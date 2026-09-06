from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, replace
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
from market_predictor.intraday.datasets import (
    prospective_analyst_revision_horizon as horizon_module,
)
from market_predictor.intraday.datasets.event_preflight import (
    load_intraday_event_preflight_config,
)
from market_predictor.intraday.datasets.prospective_analyst_revision_horizon import (
    load_prospective_analyst_revision_horizon,
    publish_prospective_analyst_revision_horizon,
)
from market_predictor.intraday.datasets.prospective_broker_actions import (
    load_prospective_broker_action_generation,
    publish_prospective_broker_action_generation,
)
from market_predictor.universe.sp500.membership_authority import (
    _membership_sha256 as membership_sha256,
)
from tests.test_intraday_prospective_broker_actions import (
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
    assert conflicted["production_available_at_utc"].isna().all()
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


def test_public_api_is_owned_by_intraday_dataset_package() -> None:
    assert (
        publish_prospective_analyst_revision_horizon.__module__
        == "market_predictor.intraday.datasets.prospective_analyst_revision_horizon"
    )
    assert (
        load_prospective_analyst_revision_horizon.__module__
        == "market_predictor.intraday.datasets.prospective_analyst_revision_horizon"
    )


def test_zero_news_horizon_has_stable_empty_schemas(tmp_path: Path) -> None:
    generation = _generation(tmp_path, "zero-news", pages=((),))
    output = tmp_path / "horizon"

    horizon = publish_prospective_analyst_revision_horizon(
        generation_directories=[generation.directory],
        output_directory=output,
        preflight_policy_path=_PREFLIGHT_POLICY,
    )

    assert horizon.classified_revisions.empty
    assert tuple(horizon.classified_revisions.columns) == horizon_module._CLASSIFIED_COLUMNS
    assert horizon.episodes.empty
    assert tuple(horizon.episodes.columns) == horizon_module._EPISODE_COLUMNS
    assert not horizon.coverage.empty
    assert tuple(horizon.coverage.columns) == horizon_module._COVERAGE_COLUMNS
    assert horizon.capacity_audit.iloc[0]["source_capacity_status"] == "blocked"
    load_prospective_analyst_revision_horizon(output)


def test_source_capacity_does_not_authorize_training_or_serving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = _generation(tmp_path, "ready-source")
    frozen = load_intraday_event_preflight_config(_PREFLIGHT_POLICY)
    source_ready = replace(
        frozen,
        minimum_unique_event_episodes=1,
        minimum_securities=1,
    )
    monkeypatch.setattr(
        horizon_module,
        "load_intraday_event_preflight_config",
        lambda _path: source_ready,
    )

    horizon = publish_prospective_analyst_revision_horizon(
        generation_directories=[generation.directory],
        output_directory=tmp_path / "horizon",
        preflight_policy_path=_PREFLIGHT_POLICY,
    )

    assert horizon.capacity_audit.iloc[0]["source_capacity_status"] == (
        "ready_for_matched_preflight"
    )
    assert horizon.capacity_audit.iloc[0]["observation_date_count"] == 1
    assert horizon.capacity_audit.iloc[0]["minimum_fit_sessions"] == 120
    assert not bool(
        horizon.capacity_audit.iloc[0]["matched_decision_capacity_evaluated"]
    )
    assert horizon.manifest["training_eligible"] is False
    assert horizon.manifest["serving_eligible"] is False
    assert horizon.authority["training_eligible"] is False
    assert horizon.authority["serving_eligible"] is False


def test_memory_guard_runs_before_parent_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = _generation(tmp_path, "memory-order")
    parent_loaded = False

    def reject_before_parent(**kwargs: object) -> None:
        if kwargs.get("stage") == "prospective analyst horizon before parent replay":
            raise DataReadinessError("memory rejected before parent replay")

    def record_parent_load(_path: Path) -> Any:
        nonlocal parent_loaded
        parent_loaded = True
        raise AssertionError("parent load must not run")

    monkeypatch.setattr(horizon_module, "assert_memory_budget", reject_before_parent)
    monkeypatch.setattr(
        horizon_module,
        "load_prospective_broker_action_generation",
        record_parent_load,
    )
    output = tmp_path / "horizon"

    with pytest.raises(DataReadinessError, match="before parent replay"):
        publish_prospective_analyst_revision_horizon(
            generation_directories=[generation.directory],
            output_directory=output,
            preflight_policy_path=_PREFLIGHT_POLICY,
        )

    assert parent_loaded is False
    assert not output.exists()


def test_memory_guards_bracket_each_parent_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = _generation(tmp_path, "memory-stages")
    stages: list[str] = []

    def record_guard(**kwargs: object) -> None:
        stages.append(str(kwargs["stage"]))

    monkeypatch.setattr(horizon_module, "assert_memory_budget", record_guard)
    publish_prospective_analyst_revision_horizon(
        generation_directories=[generation.directory],
        output_directory=tmp_path / "horizon",
        preflight_policy_path=_PREFLIGHT_POLICY,
    )

    for stage in (
        "prospective analyst horizon before generation load",
        "prospective analyst horizon after generation load",
        "prospective analyst horizon before poll load",
        "prospective analyst horizon after poll load",
    ):
        assert stages.count(stage) >= 2


def test_failed_publication_leaves_no_visible_or_staged_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = _generation(tmp_path, "failed-publication")
    output = tmp_path / "horizon"

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected canonical write failure")

    monkeypatch.setattr(horizon_module, "write_canonical_artifact", fail_write)
    with pytest.raises(RuntimeError, match="injected"):
        publish_prospective_analyst_revision_horizon(
            generation_directories=[generation.directory],
            output_directory=output,
            preflight_policy_path=_PREFLIGHT_POLICY,
        )

    assert not output.exists()
    assert not output.with_name(f".{output.name}.staging").exists()
    assert not output.with_name(f".{output.name}.staging.owner.json").exists()


def test_unowned_staging_is_preserved_and_rejected(tmp_path: Path) -> None:
    generation = _generation(tmp_path, "unowned-staging")
    output = tmp_path / "horizon"
    staging = output.with_name(f".{output.name}.staging")
    staging.mkdir()
    sentinel = staging / "unowned.txt"
    sentinel.write_text("retain\n", encoding="utf-8")

    with pytest.raises(DataReadinessError, match="not owned"):
        publish_prospective_analyst_revision_horizon(
            generation_directories=[generation.directory],
            output_directory=output,
            preflight_policy_path=_PREFLIGHT_POLICY,
        )

    assert sentinel.read_text(encoding="utf-8") == "retain\n"
    assert not output.exists()


def test_owned_staging_is_recovered_before_publication(tmp_path: Path) -> None:
    generation = _generation(tmp_path, "owned-staging")
    output = tmp_path / "horizon"
    staging = output.with_name(f".{output.name}.staging")
    owner = output.with_name(f".{output.name}.staging.owner.json")
    staging.mkdir()
    (staging / "partial.txt").write_text("partial\n", encoding="utf-8")
    _write_json(
        owner,
        {
            "schema": horizon_module.STAGING_OWNER_SCHEMA,
            "staging_directory": str(staging),
            "output_directory": str(output),
        },
    )

    publish_prospective_analyst_revision_horizon(
        generation_directories=[generation.directory],
        output_directory=output,
        preflight_policy_path=_PREFLIGHT_POLICY,
    )

    assert output.is_dir()
    assert not staging.exists()
    assert not owner.exists()
    load_prospective_analyst_revision_horizon(output)


@pytest.mark.parametrize(
    "mutation",
    ("extra_manifest", "extra_artifact_role", "bool_count", "bool_artifact_bytes"),
)
def test_loader_rejects_non_exact_manifest_shapes_and_types(
    tmp_path: Path,
    mutation: str,
) -> None:
    generation = _generation(tmp_path, f"manifest-{mutation}")
    output = tmp_path / "horizon"
    publish_prospective_analyst_revision_horizon(
        generation_directories=[generation.directory],
        output_directory=output,
        preflight_policy_path=_PREFLIGHT_POLICY,
    )
    manifest_path = output / "_manifest.json"
    manifest = _json_object(manifest_path)
    if mutation == "extra_manifest":
        manifest["unexpected"] = True
    elif mutation == "extra_artifact_role":
        manifest["artifacts"]["unexpected"] = manifest["artifacts"][
            "classified_revisions"
        ]
    elif mutation == "bool_count":
        manifest["generation_count"] = True
    else:
        manifest["artifacts"]["classified_revisions"]["bytes"] = True
    _rewrite_manifest_authority(output, manifest)

    with pytest.raises(DataReadinessError, match="schema|inventory|invalid|verify"):
        load_prospective_analyst_revision_horizon(output)


@pytest.mark.parametrize("mutation", ("duplicate_key", "non_finite", "extra_key"))
def test_loader_rejects_noncanonical_request_json(
    tmp_path: Path,
    mutation: str,
) -> None:
    generation = _generation(tmp_path, f"request-{mutation}")
    output = tmp_path / "horizon"
    publish_prospective_analyst_revision_horizon(
        generation_directories=[generation.directory],
        output_directory=output,
        preflight_policy_path=_PREFLIGHT_POLICY,
    )
    request_path = output / "_request.json"
    if mutation == "duplicate_key":
        request_path.write_text(
            '{"schema":"first","schema":"second"}\n',
            encoding="utf-8",
        )
    elif mutation == "non_finite":
        raw = request_path.read_text(encoding="utf-8")
        request_path.write_text(
            raw.replace('"memory_hard_budget_gib": 4.0', '"memory_hard_budget_gib": NaN'),
            encoding="utf-8",
        )
    else:
        request = _json_object(request_path)
        request["unexpected"] = True
        _write_json(request_path, request)

    with pytest.raises(DataReadinessError, match="JSON|request|schema"):
        load_prospective_analyst_revision_horizon(output)


@pytest.mark.parametrize("mutation", ("production_ready", "artifact_path"))
def test_loader_rejects_child_eligibility_or_path_escalation(
    tmp_path: Path,
    mutation: str,
) -> None:
    generation = _generation(tmp_path, f"child-{mutation}")
    output = tmp_path / "horizon"
    publish_prospective_analyst_revision_horizon(
        generation_directories=[generation.directory],
        output_directory=output,
        preflight_policy_path=_PREFLIGHT_POLICY,
    )
    sidecar_path = output / "classified_revisions.parquet.manifest.json"
    sidecar = _json_object(sidecar_path)
    if mutation == "production_ready":
        sidecar["production_ready"] = True
    else:
        sidecar["artifact_path"] = str(tmp_path / "different" / "classified.parquet")
    _write_json(sidecar_path, sidecar)
    manifest_path = output / "_manifest.json"
    manifest = _json_object(manifest_path)
    manifest["artifact_manifest_hashes"]["classified_revisions"] = file_sha256(
        sidecar_path
    )
    _rewrite_manifest_authority(output, manifest)

    with pytest.raises(DataReadinessError, match="lineage|child values"):
        load_prospective_analyst_revision_horizon(output)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_publication_rejects_output_windows_junction(tmp_path: Path) -> None:
    generation = _generation(tmp_path, "junction-output")
    external = tmp_path / "external-output"
    external.mkdir()
    output = tmp_path / "horizon"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(output), str(external)],
        check=False,
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip(f"junction creation unavailable: {created.stderr.strip()}")

    with pytest.raises(DataReadinessError, match="reparse point"):
        publish_prospective_analyst_revision_horizon(
            generation_directories=[generation.directory],
            output_directory=output,
            preflight_policy_path=_PREFLIGHT_POLICY,
        )
    assert not any(external.iterdir())


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_replay_rejects_root_windows_junction(tmp_path: Path) -> None:
    generation = _generation(tmp_path, "junction-replay")
    output = tmp_path / "horizon"
    publish_prospective_analyst_revision_horizon(
        generation_directories=[generation.directory],
        output_directory=output,
        preflight_policy_path=_PREFLIGHT_POLICY,
    )
    external = tmp_path / "external-horizon"
    output.rename(external)
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(output), str(external)],
        check=False,
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip(f"junction creation unavailable: {created.stderr.strip()}")

    with pytest.raises(DataReadinessError, match="reparse point"):
        load_prospective_analyst_revision_horizon(output)


def test_cross_generation_alpaca_asset_change_is_ineligible(tmp_path: Path) -> None:
    first, second = _two_generation_chain(
        tmp_path,
        "asset-change",
        first_event=_event(
            event_id="analyst-asset",
            headline="Morgan Stanley upgrades (AAA) to Buy",
        ),
        second_event=_event(
            event_id="analyst-asset",
            headline="Morgan Stanley upgrades (AAA) to Buy",
        ),
        first_asset_id="alpaca-asset-first",
        second_asset_id="alpaca-asset-second",
    )

    horizon = publish_prospective_analyst_revision_horizon(
        generation_directories=[first.directory, second.directory],
        output_directory=tmp_path / "horizon",
        preflight_policy_path=_PREFLIGHT_POLICY,
    )

    assert not horizon.classified_revisions["identity_eligible"].astype(bool).any()
    assert horizon.classified_revisions["production_available_at_utc"].isna().all()
    assert horizon.episodes.empty


def test_cross_generation_missing_asset_identity_remains_never_eligible(
    tmp_path: Path,
) -> None:
    fixture = _generation(tmp_path, "missing-asset")
    generation = load_prospective_broker_action_generation(fixture.directory)
    revisions = generation.revisions.copy()
    revisions["alpaca_asset_id"] = ""
    revisions["identity_eligible"] = False
    revisions["identity_ineligible_reason"] = "identity_never_eligible"
    revisions["identity_first_eligible_at_utc"] = pd.NaT
    revisions["production_available_at_utc"] = pd.NaT
    missing_identity = replace(
        generation,
        revisions=revisions,
    )
    merged = horizon_module._merge_revisions([missing_identity, missing_identity])

    assert set(merged["identity_ineligible_reason"].astype(str)) == {
        "identity_never_eligible"
    }
    assert not merged["identity_eligible"].astype(bool).any()
    assert merged["production_available_at_utc"].isna().all()


def test_cross_generation_provider_timestamp_collision_is_ineligible(
    tmp_path: Path,
) -> None:
    first, second = _two_generation_chain(
        tmp_path,
        "timestamp-collision",
        first_event=_event(
            event_id="analyst-collision",
            headline="Morgan Stanley upgrades (AAA) to Buy",
            content="First payload.",
            updated_at="2026-08-15T10:05:00Z",
        ),
        second_event=_event(
            event_id="analyst-collision",
            headline="Morgan Stanley raises target on (AAA)",
            content="Different payload at the same provider timestamp.",
            updated_at="2026-08-15T10:05:00Z",
        ),
    )

    horizon = publish_prospective_analyst_revision_horizon(
        generation_directories=[first.directory, second.directory],
        output_directory=tmp_path / "horizon",
        preflight_policy_path=_PREFLIGHT_POLICY,
    )

    assert len(horizon.classified_revisions) == 2
    assert not horizon.classified_revisions["identity_eligible"].astype(bool).any()
    assert horizon.classified_revisions["production_available_at_utc"].isna().all()
    assert horizon.episodes.empty


def test_event_first_seen_is_shared_across_cross_generation_revisions(
    tmp_path: Path,
) -> None:
    first, second = _two_generation_chain(
        tmp_path,
        "event-first-seen",
        first_event=_event(
            event_id="analyst-revised",
            headline="Morgan Stanley upgrades (AAA) to Buy",
            content="Initial view.",
            updated_at="2026-08-15T10:05:00Z",
        ),
        second_event=_event(
            event_id="analyst-revised",
            headline="Morgan Stanley raises target on (AAA)",
            content="Later revision.",
            updated_at="2026-08-15T10:06:00Z",
        ),
    )

    horizon = publish_prospective_analyst_revision_horizon(
        generation_directories=[first.directory, second.directory],
        output_directory=tmp_path / "horizon",
        preflight_policy_path=_PREFLIGHT_POLICY,
    )

    rows = horizon.classified_revisions.sort_values("revision_first_seen_at_utc")
    assert len(rows) == 2
    assert rows["event_first_seen_at_utc"].nunique() == 1
    assert rows["revision_first_seen_at_utc"].nunique() == 2


def test_post_rename_cleanup_failure_does_not_fail_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = _generation(tmp_path, "post-rename")
    output = tmp_path / "horizon"
    owner = output.with_name(f".{output.name}.staging.owner.json")
    original_unlink = Path.unlink

    def fail_owner_unlink(
        path: Path,
        missing_ok: bool = False,
    ) -> None:
        if path == owner and output.exists():
            raise PermissionError("injected owner cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_owner_unlink)
    horizon = publish_prospective_analyst_revision_horizon(
        generation_directories=[generation.directory],
        output_directory=output,
        preflight_policy_path=_PREFLIGHT_POLICY,
    )

    assert horizon.directory == output.resolve()
    assert output.is_dir()
    load_prospective_analyst_revision_horizon(output)


def test_zero_memory_headroom_is_rejected_at_public_boundary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="memory policy"):
        publish_prospective_analyst_revision_horizon(
            generation_directories=[],
            output_directory=tmp_path / "horizon",
            preflight_policy_path=_PREFLIGHT_POLICY,
            memory_headroom_gib=0.0,
        )


def test_missing_policy_uses_domain_error(tmp_path: Path) -> None:
    with pytest.raises(DataReadinessError, match="policy is unavailable"):
        publish_prospective_analyst_revision_horizon(
            generation_directories=[],
            output_directory=tmp_path / "horizon",
            preflight_policy_path=tmp_path / "missing-policy.toml",
        )


def test_loader_rejects_memory_evidence_above_safety_threshold(
    tmp_path: Path,
) -> None:
    generation = _generation(tmp_path, "memory-evidence")
    output = tmp_path / "horizon"
    publish_prospective_analyst_revision_horizon(
        generation_directories=[generation.directory],
        output_directory=output,
        preflight_policy_path=_PREFLIGHT_POLICY,
    )
    manifest = _json_object(output / "_manifest.json")
    threshold = float(manifest["memory"]["safety_threshold_gib"])
    manifest["memory"]["current_working_set_gib"] = threshold + 0.01
    manifest["memory"]["peak_working_set_gib"] = threshold + 0.02
    _rewrite_manifest_authority(output, manifest)

    with pytest.raises(DataReadinessError, match="memory record"):
        load_prospective_analyst_revision_horizon(output)


@pytest.mark.parametrize(
    "mutation",
    ("extra_key", "duplicate_key", "bool_rows", "malformed_audit"),
)
def test_loader_rejects_noncanonical_child_manifest_json(
    tmp_path: Path,
    mutation: str,
) -> None:
    generation = _generation(tmp_path, f"child-json-{mutation}")
    output = tmp_path / "horizon"
    publish_prospective_analyst_revision_horizon(
        generation_directories=[generation.directory],
        output_directory=output,
        preflight_policy_path=_PREFLIGHT_POLICY,
    )
    sidecar_path = output / "classified_revisions.parquet.manifest.json"
    sidecar = _json_object(sidecar_path)
    if mutation == "extra_key":
        sidecar["unexpected"] = True
        _write_json(sidecar_path, sidecar)
    elif mutation == "duplicate_key":
        raw = sidecar_path.read_text(encoding="utf-8")
        sidecar_path.write_text(
            raw.replace(
                '"schema":',
                '"schema": "duplicate",\n  "schema":',
                1,
            ),
            encoding="utf-8",
        )
    elif mutation == "bool_rows":
        sidecar["rows"] = True
        _write_json(sidecar_path, sidecar)
    else:
        sidecar["audit"] = [
            {
                "name": "classified_revisions",
                "status": "pass",
                "failures": False,
                "rows_checked": sidecar["rows"],
                "detail": "malformed failures type",
            }
        ]
        _write_json(sidecar_path, sidecar)
    manifest = _json_object(output / "_manifest.json")
    manifest["artifact_manifest_hashes"]["classified_revisions"] = file_sha256(
        sidecar_path
    )
    _rewrite_manifest_authority(output, manifest)

    with pytest.raises(DataReadinessError, match="JSON|child|rows|failures"):
        load_prospective_analyst_revision_horizon(output)


def test_projected_derivation_memory_rejects_before_revision_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = _generation(tmp_path, "derivation-capacity")
    materialized = False

    def unexpected_merge(_generations: object) -> pd.DataFrame:
        nonlocal materialized
        materialized = True
        raise AssertionError("revision materialization must not run")

    monkeypatch.setattr(
        horizon_module,
        "DERIVATION_EXPANSION_FACTOR",
        10**12,
    )
    monkeypatch.setattr(horizon_module, "_merge_revisions", unexpected_merge)

    with pytest.raises(DataReadinessError, match="projected derivation memory"):
        publish_prospective_analyst_revision_horizon(
            generation_directories=[generation.directory],
            output_directory=tmp_path / "horizon",
            preflight_policy_path=_PREFLIGHT_POLICY,
        )
    assert materialized is False


def _rewrite_manifest_authority(
    output: Path,
    manifest: dict[str, Any],
) -> None:
    manifest_path = output / "_manifest.json"
    _write_json(manifest_path, manifest)
    authority_path = output / "_authority.json"
    authority = _json_object(authority_path)
    authority["artifact_sha256"] = file_sha256(manifest_path)
    _write_json(authority_path, authority)


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


def _two_generation_chain(
    tmp_path: Path,
    name: str,
    *,
    first_event: dict[str, object],
    second_event: dict[str, object],
    first_asset_id: str = "alpaca-asset-aaa",
    second_asset_id: str = "alpaca-asset-aaa",
) -> tuple[_GenerationFixture, _GenerationFixture]:
    root = tmp_path / name
    root.mkdir()
    membership = _membership_authority(root)
    registry = root / "poll-registry"
    first_at = OBSERVED_AT
    second_at = OBSERVED_AT + timedelta(minutes=1)

    def collect(
        *,
        poll: Path,
        observed_at: datetime,
        event: dict[str, object],
        asset_id: str,
        previous: Path | None,
    ) -> None:
        received_seconds = int((observed_at - OBSERVED_AT).total_seconds()) + 2
        collect_prospective_broker_action_poll(
            membership_authority_directory=membership,
            registry_directory=registry,
            output_directory=poll,
            fetch_assets=lambda: _assets_with_id(asset_id),
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
        poll=first_poll,
        observed_at=first_at,
        event=first_event,
        asset_id=first_asset_id,
        previous=None,
    )
    collect(
        poll=second_poll,
        observed_at=second_at,
        event=second_event,
        asset_id=second_asset_id,
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
