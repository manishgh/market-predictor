from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

import market_predictor.edge_rebuild.sp500_memberships as membership_module
from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.universe import (
    IndexChange,
    IndexChangeSource,
    VerifiedIndexChanges,
)


def test_fox_event_resolution_takes_one_hop_per_atomic_timestamp() -> None:
    moment = pd.Timestamp("2019-03-19T04:00:00Z")
    transitions = _transitions(
        [
            ("FOX", "TFCF"),
            ("FOXA", "TFCFA"),
            ("FOXAV", "FOXA"),
            ("FOXBV", "FOX"),
        ],
        moment=moment,
    )
    change = _change(
        action="addition",
        ticker="FOXBV",
        effective_at=moment.to_pydatetime(),
        published=date(2019, 3, 14),
    )

    resolved = membership_module._resolved_change(change, transitions)

    assert resolved.ticker == "FOX"


def test_anchor_event_mismatch_excludes_complete_security_below_cap() -> None:
    anchor = _anchor(500)
    missing_addition = _change(
        action="addition",
        ticker="MISS",
        effective_at=datetime(2025, 1, 2, 5, tzinfo=UTC),
        published=date(2024, 12, 20),
    )

    memberships, exclusions, audit = membership_module._build_memberships(
        anchor=anchor,
        changes=[missing_addition],
        transitions=_transitions([]),
        start_date=date(2024, 1, 2),
        cutoff_date=date(2026, 7, 8),
        security_exclusions_path=None,
        maximum_security_exclusion_fraction=0.05,
        snapshot_id="snapshot",
    )

    assert len(memberships) == 500
    assert exclusions[0]["ticker"] == "MISS"
    assert exclusions[0]["reason"] == "addition_absent_from_cutoff_anchor_replay"
    assert audit["excluded_security_count"] == 1
    assert audit["benchmark_session_exclusions"] == 0


def test_more_than_five_percent_security_exclusions_fail() -> None:
    memberships = pd.DataFrame({"security_id": [f"sec:{number}" for number in range(100)]})
    automatic = [
        {
            "security_id": f"sec:{number}",
            "ticker": f"T{number}",
            "reason": "poison",
            "effective_at_utc": "2025-01-01T05:00:00+00:00",
        }
        for number in range(6)
    ]

    with pytest.raises(DataReadinessError, match="above the frozen 5.00% ceiling"):
        membership_module._apply_security_exclusions(
            memberships,
            automatic_exclusions=automatic,
            security_exclusions_path=None,
            maximum_security_exclusion_fraction=0.05,
        )


def test_benchmark_exclusion_is_refused(tmp_path: Path) -> None:
    memberships = pd.DataFrame({"security_id": ["sec:one"]})
    exclusions = tmp_path / "exclusions.csv"
    exclusions.write_text(
        "security_id,ticker,reason\nsec:one,SPY,poison\n",
        encoding="utf-8",
    )

    with pytest.raises(DataReadinessError, match="benchmark sessions cannot be excluded"):
        membership_module._apply_security_exclusions(
            memberships,
            automatic_exclusions=[],
            security_exclusions_path=exclusions,
            maximum_security_exclusion_fraction=0.05,
        )


def test_anchor_poison_invalidates_published_membership_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "raw"
    events = tmp_path / "events"
    transitions = tmp_path / "transitions"
    for directory, payload in (
        (archive, {"artifact_sha256": "a" * 64}),
        (events, {"event_set_sha256": "b" * 64}),
        (transitions, {"transition_set_sha256": "c" * 64}),
    ):
        directory.mkdir()
        membership_module._write_json_atomic(directory / "_authority.json", payload)
    reviewed = tmp_path / "reviewed.csv"
    reviewed.write_text("bound input\n", encoding="utf-8")
    anchor_path = tmp_path / "anchor.csv"
    _anchor(500).to_csv(anchor_path, index=False)
    empty_transitions = _transitions([])
    verified_events = VerifiedIndexChanges(
        changes=(),
        authority_sha256="d" * 64,
        event_set_sha256="e" * 64,
    )
    monkeypatch.setattr(
        membership_module,
        "require_sp500_transition_authority",
        lambda *_, **__: empty_transitions,
    )
    monkeypatch.setattr(
        membership_module,
        "require_spglobal_event_reconstruction_ready",
        lambda *_, **__: verified_events,
    )
    output = tmp_path / "memberships"
    membership_module.publish_sp500_membership_authority(
        archive_directory=archive,
        event_directory=events,
        transition_directory=transitions,
        reviewed_transitions_path=reviewed,
        anchor_path=anchor_path,
        start_date=date(2018, 5, 29),
        cutoff_date=date(2026, 7, 8),
        output_directory=output,
    )
    poisoned = pd.read_csv(anchor_path)
    poisoned.loc[0, "company"] = "POISONED"
    poisoned.to_csv(anchor_path, index=False)

    with pytest.raises(DataReadinessError, match="request identity"):
        membership_module.require_sp500_membership_authority(
            output,
            archive_directory=archive,
            event_directory=events,
            transition_directory=transitions,
            reviewed_transitions_path=reviewed,
            anchor_path=anchor_path,
            start_date=date(2018, 5, 29),
            cutoff_date=date(2026, 7, 8),
        )


def _anchor(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [f"T{number:03d}" for number in range(count)],
            "company": [f"Company {number}" for number in range(count)],
            "sector": ["Industrials"] * count,
            "industry": ["Machinery"] * count,
            "cik": [f"{number + 1:010d}" for number in range(count)],
        }
    )


def _change(
    *,
    action: str,
    ticker: str,
    effective_at: datetime,
    published: date,
) -> IndexChange:
    source = IndexChangeSource(
        source_url="https://press.spglobal.com/test",
        source_published_date=published,
        source_sha256="a" * 64,
    )
    return IndexChange(
        effective_at_utc=effective_at,
        action=action,
        ticker=ticker,
        company=f"{ticker} Company",
        sector="Industrials",
        source_url=source.source_url,
        source_published_date=published,
        source_sha256=source.source_sha256,
        supporting_sources=(source,),
    )


def _transitions(
    pairs: list[tuple[str, str]],
    *,
    moment: pd.Timestamp = pd.Timestamp("2020-01-02T05:00:00Z"),
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transition_id": [f"transition-{number}" for number in range(len(pairs))],
            "effective_at_utc": pd.Series(
                [moment] * len(pairs),
                dtype="datetime64[ns, UTC]",
            ),
            "old_ticker": [pair[0] for pair in pairs],
            "new_ticker": [pair[1] for pair in pairs],
            "identity_continuity": [True] * len(pairs),
            "membership_continuity": [True] * len(pairs),
            "old_security_id": [""] * len(pairs),
            "new_security_id": [""] * len(pairs),
            "source_url": ["https://press.spglobal.com/test"] * len(pairs),
        }
    )
