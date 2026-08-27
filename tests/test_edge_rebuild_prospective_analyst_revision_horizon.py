from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.prospective_analyst_revision_horizon import (
    load_prospective_analyst_revision_horizon,
    publish_prospective_analyst_revision_horizon,
)
from market_predictor.edge_rebuild.prospective_broker_actions import (
    publish_prospective_broker_action_generation,
)
from market_predictor.universe.sp500.membership_authority import (
    _membership_sha256 as membership_sha256,
)
from tests.test_edge_rebuild_prospective_broker_actions import (
    OBSERVED_AT,
    _a43_dataset,
    _assets_with_id,
    _Clock,
    _membership_authority,
    _news,
    _page,
    _passing_audit,
    collect_prospective_broker_action_poll,
)

_ROOT = Path(__file__).parents[1]
_PREFLIGHT_POLICY = _ROOT / "configs" / "edge_rebuild_intraday_event_preflight.toml"


@dataclass(frozen=True, slots=True)
class _GenerationFixture:
    directory: Path
    poll_directories: tuple[Path, ...]


def test_multiple_revisions_count_as_one_exact_analyst_episode(
    tmp_path: Path,
) -> None:
    generation = _generation(
        tmp_path,
        "multi-revision",
        pages=(
            (
                _event(
                    event_id="analyst-1",
                    headline="Morgan Stanley upgrades (AAA) to Buy",
                    content="The broker upgraded the shares.",
                    updated_at="2026-08-15T10:05:00Z",
                ),
                _event(
                    event_id="earnings-1",
                    headline="(AAA) reports Q2 earnings",
                    content="The company reported quarterly results.",
                    updated_at="2026-08-15T10:06:00Z",
                ),
            ),
            (
                _event(
                    event_id="analyst-1",
                    headline="Morgan Stanley raises price target on (AAA)",
                    content="The broker corrected and raised its price target.",
                    updated_at="2026-08-15T10:10:00Z",
                ),
            ),
        ),
    )

    output = tmp_path / "horizon"
    publish_prospective_analyst_revision_horizon(
        generation_directories=[generation.directory],
        output_directory=output,
        preflight_policy_path=_PREFLIGHT_POLICY,
    )
    horizon = load_prospective_analyst_revision_horizon(output)

    analyst_revisions = horizon.classified_revisions.loc[
        horizon.classified_revisions["provider_event_id"].eq("analyst-1")
    ]
    assert len(analyst_revisions) == 2
    assert analyst_revisions["revision_id"].nunique() == 2
    assert analyst_revisions["classified_analyst_revision"].astype(bool).all()
    assert horizon.episodes["provider_event_id"].tolist() == ["analyst-1"]
    assert horizon.episodes["revision_count"].tolist() == [2]
    assert "earnings-1" not in set(horizon.episodes["provider_event_id"])


def test_non_analyst_events_are_retained_but_never_admitted(
    tmp_path: Path,
) -> None:
    generation = _generation(
        tmp_path,
        "classification",
        pages=(
            (
                _event(
                    event_id="analyst-1",
                    headline="Morgan Stanley downgrades (AAA) to Sell",
                ),
                _event(
                    event_id="earnings-1",
                    headline="(AAA) reports Q2 earnings",
                ),
                _event(
                    event_id="unclassified-1",
                    headline="(AAA) schedules its annual shareholder meeting",
                ),
            ),
        ),
    )

    output = tmp_path / "horizon"
    publish_prospective_analyst_revision_horizon(
        generation_directories=[generation.directory],
        output_directory=output,
        preflight_policy_path=_PREFLIGHT_POLICY,
    )
    horizon = load_prospective_analyst_revision_horizon(output)

    assert set(horizon.classified_revisions["provider_event_id"]) == {
        "analyst-1",
        "earnings-1",
        "unclassified-1",
    }
    assert horizon.episodes["provider_event_id"].tolist() == ["analyst-1"]
    non_analyst = horizon.classified_revisions.loc[
        ~horizon.classified_revisions["provider_event_id"].eq("analyst-1")
    ]
    assert not non_analyst["classified_analyst_revision"].astype(bool).any()


def test_horizon_is_capacity_evidence_only(tmp_path: Path) -> None:
    generation = _generation(tmp_path, "capacity-only")
    output = tmp_path / "horizon"

    publish_prospective_analyst_revision_horizon(
        generation_directories=[generation.directory],
        output_directory=output,
        preflight_policy_path=_PREFLIGHT_POLICY,
    )
    horizon = load_prospective_analyst_revision_horizon(output)

    assert horizon.manifest["training_eligible"] is False
    assert horizon.manifest["serving_eligible"] is False
    assert horizon.manifest["future_holdout_opened"] is False
    assert horizon.authority["training_eligible"] is False
    assert horizon.authority["serving_eligible"] is False
    assert horizon.authority["future_holdout_opened"] is False
    assert isinstance(horizon.coverage, pd.DataFrame)
    assert isinstance(horizon.capacity_audit, pd.DataFrame)


@pytest.mark.parametrize("duplicate_kind", ("generation", "poll"))
def test_duplicate_generation_or_poll_is_rejected(
    tmp_path: Path,
    duplicate_kind: str,
) -> None:
    generation = _generation(tmp_path, "duplicate-parent")
    if duplicate_kind == "generation":
        parents = [generation.directory, generation.directory]
    else:
        duplicate = tmp_path / "duplicate-generation"
        publish_prospective_broker_action_generation(
            poll_directories=generation.poll_directories,
            output_directory=duplicate,
        )
        parents = [generation.directory, duplicate]

    with pytest.raises(DataReadinessError, match="duplicate|overlap|poll"):
        publish_prospective_analyst_revision_horizon(
            generation_directories=parents,
            output_directory=tmp_path / f"rejected-{duplicate_kind}",
            preflight_policy_path=_PREFLIGHT_POLICY,
        )


def test_cross_generation_security_conflict_is_ineligible(
    tmp_path: Path,
) -> None:
    first, second = _conflicting_security_generations(tmp_path)
    output = tmp_path / "horizon"

    publish_prospective_analyst_revision_horizon(
        generation_directories=[first.directory, second.directory],
        output_directory=output,
        preflight_policy_path=_PREFLIGHT_POLICY,
    )
    horizon = load_prospective_analyst_revision_horizon(output)

    conflicted = horizon.classified_revisions.loc[
        horizon.classified_revisions["provider_event_id"].eq("analyst-1")
    ]
    assert not conflicted["identity_eligible"].astype(bool).any()
    assert conflicted["eligibility_reason"].astype(str).str.contains(
        "security|identity|conflict|changed",
        case=False,
    ).all()
    assert horizon.episodes.empty
    assert horizon.coverage["previous_poll_at_utc"].isna().any()
    assert horizon.coverage["previous_poll_at_utc"].notna().any()


@pytest.mark.parametrize("mutation", ("tamper", "extra"))
def test_strict_loader_rejects_tamper_and_extra_files(
    tmp_path: Path,
    mutation: str,
) -> None:
    generation = _generation(tmp_path, f"strict-{mutation}")
    output = tmp_path / "horizon"
    publish_prospective_analyst_revision_horizon(
        generation_directories=[generation.directory],
        output_directory=output,
        preflight_policy_path=_PREFLIGHT_POLICY,
    )
    load_prospective_analyst_revision_horizon(output)

    if mutation == "tamper":
        artifacts = sorted(output.glob("*.parquet"))
        assert artifacts
        artifacts[0].write_bytes(artifacts[0].read_bytes() + b"tampered")
    else:
        (output / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(DataReadinessError, match="hash|integrity|inventory|unexpected"):
        load_prospective_analyst_revision_horizon(output)


def test_first_seen_cannot_be_replaced_by_provider_publication_time(
    tmp_path: Path,
) -> None:
    published_at = pd.Timestamp("2020-01-01T10:00:00Z")
    generation = _generation(
        tmp_path,
        "observed-availability",
        pages=(
            (
                _event(
                    event_id="analyst-1",
                    headline="Morgan Stanley upgrades (AAA) to Buy",
                    created_at=published_at.isoformat(),
                    updated_at="2020-01-01T10:05:00Z",
                ),
            ),
        ),
    )
    parent = load_canonical_artifact(
        generation.directory / "event_revisions.parquet",
        expected_type="prospective_broker_action_revisions",
        allow_research=True,
    )[0].iloc[0]
    output = tmp_path / "horizon"

    publish_prospective_analyst_revision_horizon(
        generation_directories=[generation.directory],
        output_directory=output,
        preflight_policy_path=_PREFLIGHT_POLICY,
    )
    revision = load_prospective_analyst_revision_horizon(
        output
    ).classified_revisions.iloc[0]

    expected_first_seen = pd.Timestamp(parent["event_first_seen_at_utc"])
    assert expected_first_seen > published_at
    assert pd.Timestamp(revision["event_first_seen_at_utc"]) == expected_first_seen
    assert pd.Timestamp(revision["production_available_at_utc"]) >= expected_first_seen
    assert pd.Timestamp(revision["event_first_seen_at_utc"]) != pd.Timestamp(
        revision["published_at_utc"]
    )


def test_policy_cannot_raise_memory_budget_above_four_gib(tmp_path: Path) -> None:
    generation = _generation(tmp_path, "memory-policy")
    policy = tmp_path / "weakened-preflight.toml"
    policy.write_text(
        _PREFLIGHT_POLICY.read_text(encoding="utf-8").replace(
            "maximum_process_memory_gib = 4.0",
            "maximum_process_memory_gib = 4.5",
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataReadinessError, match="memory|frozen|4"):
        publish_prospective_analyst_revision_horizon(
            generation_directories=[generation.directory],
            output_directory=tmp_path / "rejected-memory-policy",
            preflight_policy_path=policy,
        )


def _generation(
    tmp_path: Path,
    name: str,
    *,
    pages: tuple[tuple[dict[str, object], ...], ...] | None = None,
    observed_at: datetime = OBSERVED_AT,
    security_id: str = "security:aaa",
) -> _GenerationFixture:
    root = tmp_path / name
    root.mkdir()
    membership = _membership_with_security_id(root, security_id)
    poll = root / "poll"
    page_rows = pages or (
        (
            _event(
                event_id="analyst-1",
                headline="Morgan Stanley upgrades (AAA) to Buy",
            ),
        ),
    )
    offset_seconds = int((observed_at - OBSERVED_AT).total_seconds())

    def fetch_page(
        _symbols: str,
        _start: datetime,
        _end: datetime,
        token: str | None,
    ) -> Any:
        index = 0 if token is None else int(token.rsplit("-", maxsplit=1)[1]) - 1
        assert 0 <= index < len(page_rows)
        next_token = f"page-{index + 2}" if index + 1 < len(page_rows) else None
        return _page(
            token,
            next_token,
            page_rows[index],
            received_seconds=offset_seconds + index + 2,
        )

    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=poll,
        fetch_assets=lambda: _assets_with_id(f"alpaca-asset-{name}"),
        fetch_page=fetch_page,
        observed_at_utc=observed_at,
        clock=_Clock(observed_at),
    )
    generation = root / "generation"
    publish_prospective_broker_action_generation(
        poll_directories=[poll],
        output_directory=generation,
    )
    return _GenerationFixture(
        directory=generation,
        poll_directories=(poll,),
    )


def _membership_with_security_id(root: Path, security_id: str) -> Path:
    membership = _membership_authority(root)
    if security_id == "security:aaa":
        return membership

    path = membership / "memberships.parquet"
    frame = load_canonical_artifact(
        path,
        expected_type="memberships",
        allow_research=True,
    )[0]
    frame.loc[:, "security_id"] = security_id
    _rewrite_membership(membership, frame)
    return membership


def _conflicting_security_generations(
    tmp_path: Path,
) -> tuple[_GenerationFixture, _GenerationFixture]:
    root = tmp_path / "security-conflict"
    root.mkdir()
    first_membership = _membership_authority(
        root,
        cutoff_date="2026-08-14",
    )
    first_frame = load_canonical_artifact(
        first_membership / "memberships.parquet",
        expected_type="memberships",
        allow_research=True,
    )[0]
    first_frame.loc[:, "security_id"] = "security:aaa:first"
    _rewrite_membership(first_membership, first_frame)
    a43 = _a43_dataset(first_membership)

    second_membership = _membership_authority(
        root,
        cutoff_date="2026-08-15",
    )
    transition = pd.Timestamp("2026-08-15T00:00:00Z")
    second_frame = first_frame.copy()
    second_frame["effective_to_utc"] = pd.Series(
        [transition] * len(second_frame),
        dtype="datetime64[ns, UTC]",
    )
    successor = second_frame.iloc[0].copy()
    successor["security_id"] = "security:aaa:second"
    successor["effective_from_utc"] = transition
    successor["effective_to_utc"] = pd.NaT
    successor["available_at_utc"] = transition
    second_frame = pd.concat(
        [second_frame, successor.to_frame().T],
        ignore_index=True,
    )
    _rewrite_membership(second_membership, second_frame)

    registry = root / "poll-registry"
    first_at = datetime.fromisoformat("2026-08-15T04:00:00+00:00")
    second_at = first_at + timedelta(minutes=1)
    event = _event(
        event_id="analyst-1",
        headline="Morgan Stanley upgrades (AAA) to Buy",
        created_at="2026-08-15T03:30:00Z",
        updated_at="2026-08-15T03:35:00Z",
    )

    def collect(
        *,
        membership: Path,
        poll: Path,
        observed_at: datetime,
        previous: Path | None,
    ) -> None:
        received_seconds = int((observed_at - OBSERVED_AT).total_seconds()) + 2
        collect_prospective_broker_action_poll(
            membership_authority_directory=membership,
            intraday_bar_dataset_directory=a43,
            registry_directory=registry,
            output_directory=poll,
            fetch_assets=lambda: _assets_with_id("alpaca-asset-conflict"),
            fetch_page=lambda *_: _page(
                None,
                None,
                (event,),
                received_seconds=received_seconds,
            ),
            observed_at_utc=observed_at,
            previous_poll_directory=previous,
            clock=_Clock(observed_at),
        )

    first_poll = root / "poll-first"
    second_poll = root / "poll-second"
    collect(
        membership=first_membership,
        poll=first_poll,
        observed_at=first_at,
        previous=None,
    )
    collect(
        membership=second_membership,
        poll=second_poll,
        observed_at=second_at,
        previous=first_poll,
    )
    first_generation = root / "generation-first"
    second_generation = root / "generation-second"
    publish_prospective_broker_action_generation(
        poll_directories=[first_poll],
        output_directory=first_generation,
    )
    publish_prospective_broker_action_generation(
        poll_directories=[second_poll],
        output_directory=second_generation,
    )
    return (
        _GenerationFixture(first_generation, (first_poll,)),
        _GenerationFixture(second_generation, (second_poll,)),
    )


def _rewrite_membership(membership: Path, frame: pd.DataFrame) -> None:
    path = membership / "memberships.parquet"
    child = load_canonical_artifact(
        path,
        expected_type="memberships",
        allow_research=True,
    )[1]
    write_canonical_artifact(
        frame,
        path,
        artifact_type="memberships",
        audit=_passing_audit("membership_fixture", len(frame)),
        inputs=child["inputs"],
        production_ready=False,
    )

    manifest_path = membership / "_manifest.json"
    manifest = _json_object(manifest_path)
    universe_sha256 = membership_sha256(frame)
    manifest["universe_sha256"] = universe_sha256
    manifest["membership_artifact"] = {
        "path": path.name,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    manifest["membership_manifest_sha256"] = file_sha256(manifest_path_for(path))
    manifest["membership_intervals"] = len(frame)
    manifest["security_count"] = frame["security_id"].nunique()
    manifest["ticker_count"] = frame["ticker"].nunique()
    _write_json(manifest_path, manifest)

    authority_path = membership / "_authority.json"
    authority = _json_object(authority_path)
    authority["artifact_sha256"] = file_sha256(manifest_path)
    authority["universe_sha256"] = universe_sha256
    authority["membership_intervals"] = len(frame)
    authority["security_count"] = frame["security_id"].nunique()
    _write_json(authority_path, authority)


def _event(
    *,
    event_id: str,
    headline: str,
    content: str = "Broker changes its view.",
    created_at: str = "2026-08-15T10:00:00Z",
    updated_at: str = "2026-08-15T10:05:00Z",
) -> dict[str, object]:
    event = _news(
        headline=headline,
        content=content,
        updated_at=updated_at,
    )
    event["id"] = event_id
    event["created_at"] = created_at
    event["url"] = f"https://example.test/news/{event_id}"
    return event


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
