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


def test_extension_preserves_base_prefix_and_allows_later_metadata() -> None:
    base_anchor = _anchor(500)
    base, _, _ = membership_module._build_memberships(
        anchor=base_anchor,
        changes=[],
        transitions=_transitions([]),
        start_date=date(2024, 1, 2),
        cutoff_date=date(2026, 7, 8),
        security_exclusions_path=None,
        maximum_security_exclusion_fraction=0.05,
        snapshot_id="base",
    )
    current_anchor = base_anchor.iloc[1:].copy()
    current_anchor.loc[current_anchor["ticker"].eq("T001"), "sector"] = "Health Care"
    current_anchor.loc[current_anchor["ticker"].eq("T001"), "industry"] = "Biotechnology"
    current_anchor = pd.concat(
        [
            current_anchor,
            pd.DataFrame(
                {
                    "ticker": ["NEW"],
                    "company": ["New Company"],
                    "sector": ["Industrials"],
                    "industry": ["Machinery"],
                    "cik": ["9999999999"],
                }
            ),
        ],
        ignore_index=True,
    )
    effective_at = datetime(2026, 8, 5, 4, tzinfo=UTC)

    current, _, _ = membership_module._build_memberships(
        anchor=current_anchor,
        changes=[
            _change(
                action="addition",
                ticker="NEW",
                effective_at=effective_at,
                published=date(2026, 7, 31),
            ),
            _change(
                action="deletion",
                ticker="T000",
                effective_at=effective_at,
                published=date(2026, 7, 31),
            ),
        ],
        transitions=_transitions([]),
        start_date=date(2024, 1, 2),
        cutoff_date=date(2026, 8, 15),
        security_exclusions_path=None,
        maximum_security_exclusion_fraction=0.05,
        snapshot_id="extension",
        base_memberships=base,
        base_cutoff_date=date(2026, 7, 8),
    )

    membership_module.verify_membership_namespace_extension(
        base,
        current,
        base_cutoff_date="2026-07-08",
        current_cutoff_date="2026-08-15",
    )
    assert set(current.loc[current["ticker"].eq("T000"), "security_id"]) == {"cik:0000000001"}
    assert set(current.loc[current["ticker"].eq("T001"), "security_id"]) == {"cik:0000000002"}
    t001 = current.loc[current["ticker"].eq("T001")].sort_values(
        "effective_from_utc"
    )
    assert list(t001["sector"]) == ["Industrials", "Health Care"]
    poisoned = current.copy()
    historical = poisoned["effective_from_utc"].lt(
        pd.Timestamp("2026-07-09T00:00:00Z")
    )
    poisoned.loc[historical & poisoned["ticker"].eq("T001"), "sector"] = "Energy"
    with pytest.raises(DataReadinessError, match="base identity namespace"):
        membership_module.verify_membership_namespace_extension(
            base,
            poisoned,
            base_cutoff_date="2026-07-08",
            current_cutoff_date="2026-08-15",
        )


def test_extension_rejects_same_ticker_with_conflicting_cik() -> None:
    base_anchor = _anchor(500)
    base, _, _ = membership_module._build_memberships(
        anchor=base_anchor,
        changes=[],
        transitions=_transitions([]),
        start_date=date(2024, 1, 2),
        cutoff_date=date(2026, 7, 8),
        security_exclusions_path=None,
        maximum_security_exclusion_fraction=0.05,
        snapshot_id="base",
    )
    current_anchor = base_anchor.copy()
    current_anchor.loc[current_anchor["ticker"].eq("T001"), "cik"] = "9999999998"

    with pytest.raises(DataReadinessError, match="CIK conflicts"):
        membership_module._build_memberships(
            anchor=current_anchor,
            changes=[],
            transitions=_transitions([]),
            start_date=date(2024, 1, 2),
            cutoff_date=date(2026, 8, 15),
            security_exclusions_path=None,
            maximum_security_exclusion_fraction=0.05,
            snapshot_id="extension",
            base_memberships=base,
            base_cutoff_date=date(2026, 7, 8),
        )


def test_extension_keeps_current_ticker_for_cik_continuous_transition() -> None:
    base_anchor = _anchor(500)
    base, _, _ = membership_module._build_memberships(
        anchor=base_anchor,
        changes=[],
        transitions=_transitions([]),
        start_date=date(2024, 1, 2),
        cutoff_date=date(2026, 7, 8),
        security_exclusions_path=None,
        maximum_security_exclusion_fraction=0.05,
        snapshot_id="base",
    )
    current_anchor = base_anchor.copy()
    current_anchor.loc[current_anchor["ticker"].eq("T001"), "ticker"] = "NEW"
    transition_at = pd.Timestamp("2026-08-05T04:00:00Z")

    current, _, _ = membership_module._build_memberships(
        anchor=current_anchor,
        changes=[],
        transitions=_transitions([("T001", "NEW")], moment=transition_at),
        start_date=date(2024, 1, 2),
        cutoff_date=date(2026, 8, 15),
        security_exclusions_path=None,
        maximum_security_exclusion_fraction=0.05,
        snapshot_id="extension",
        base_memberships=base,
        base_cutoff_date=date(2026, 7, 8),
    )

    active = current[
        current["effective_to_utc"].isna()
        & current["ticker"].eq("NEW")
    ]
    assert set(active["security_id"]) == {"cik:0000000002"}
    membership_module.verify_membership_namespace_extension(
        base,
        current,
        base_cutoff_date="2026-07-08",
        current_cutoff_date="2026-08-15",
    )


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
    original_manifest = membership_module._load_object(output / "_manifest.json")
    original_authority = membership_module._load_object(output / "_authority.json")
    for field, value in (
        ("parent_lineage", {"tampered": "lineage"}),
        ("cutoff_date", "2026-07-07"),
    ):
        poisoned_manifest = {**original_manifest, field: value}
        membership_module._write_json_atomic(
            output / "_manifest.json",
            poisoned_manifest,
        )
        poisoned_authority = {
            **original_authority,
            "artifact_sha256": membership_module.file_sha256(
                output / "_manifest.json"
            ),
        }
        membership_module._write_json_atomic(
            output / "_authority.json",
            poisoned_authority,
        )
        with pytest.raises(DataReadinessError, match="base S&P membership authority"):
            membership_module._load_extension_parent(
                output,
                start_date=date(2018, 5, 29),
                cutoff_date=date(2026, 8, 15),
            )
    membership_module._write_json_atomic(
        output / "_manifest.json",
        original_manifest,
    )
    membership_module._write_json_atomic(
        output / "_authority.json",
        original_authority,
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
