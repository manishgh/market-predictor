from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import pyarrow.parquet as pq
import pytest

from market_predictor.canonical.audits import (
    CanonicalAuditCheck,
    CanonicalAuditReport,
)
from market_predictor.canonical.store import file_sha256, write_canonical_artifact
from market_predictor.edge_rebuild.intraday_bar_dataset import (
    _arrow_schema_record,
    _transformation_identity,
)
from market_predictor.edge_rebuild.intraday_history import json_sha256
from market_predictor.edge_rebuild.prospective_broker_actions import (
    _verify_asset_request_url,
    load_prospective_broker_action_generation,
    load_prospective_broker_action_poll,
    publish_prospective_broker_action_generation,
)
from market_predictor.edge_rebuild.prospective_broker_actions import (
    collect_prospective_broker_action_poll as _collect_prospective_poll,
)
from market_predictor.sources.alpaca import AlpacaAssetSnapshot, AlpacaNewsPage
from market_predictor.v3.errors import DataReadinessError

OBSERVED_AT = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)


def test_asset_request_accepts_exact_paper_trading_host() -> None:
    url = (
        "https://paper-api.alpaca.markets/v2/assets?"
        + urlencode({"status": "active", "asset_class": "us_equity"})
    )

    _verify_asset_request_url(url, final_url=url, redirect_chain=())


def test_asset_request_rejects_unapproved_host() -> None:
    url = (
        "https://example.com/v2/assets?"
        + urlencode({"status": "active", "asset_class": "us_equity"})
    )

    with pytest.raises(DataReadinessError, match="approved Alpaca asset host"):
        _verify_asset_request_url(url, final_url=url, redirect_chain=())


class _Clock:
    def __init__(self, start: datetime = OBSERVED_AT) -> None:
        self._value = start

    def __call__(self) -> datetime:
        self._value += timedelta(seconds=1)
        return self._value


def collect_prospective_broker_action_poll(**kwargs: Any) -> dict[str, object]:
    membership = Path(kwargs["membership_authority_directory"])
    kwargs.setdefault(
        "intraday_bar_dataset_directory", _a43_dataset(membership)
    )
    kwargs.setdefault("registry_directory", membership.parent / "poll-registry")
    fetch_assets = kwargs["fetch_assets"]
    fetch_page = kwargs["fetch_page"]
    scheduled = kwargs.get("observed_at_utc", OBSERVED_AT)
    assert isinstance(scheduled, datetime)

    def bound_assets() -> AlpacaAssetSnapshot:
        snapshot = fetch_assets()
        url = (
            "https://api.alpaca.markets/v2/assets?"
            + urlencode({"status": "active", "asset_class": "us_equity"})
        )
        return replace(
            snapshot,
            requested_url=url,
            final_url=url,
            redirect_chain=(),
            retrieved_at_utc=scheduled + timedelta(seconds=1),
        )

    def bound_page(
        symbols: str,
        start: datetime,
        end: datetime,
        token: str | None,
    ) -> AlpacaNewsPage:
        page = fetch_page(symbols, start, end, token)
        params = {
            "symbols": symbols,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "sort": "asc",
            "limit": "50",
            "include_content": "true",
        }
        if token is not None:
            params["page_token"] = token
        url = "https://data.alpaca.markets/v1beta1/news?" + urlencode(params)
        return replace(page, requested_url=url, final_url=url, redirect_chain=())

    kwargs["fetch_assets"] = bound_assets
    kwargs["fetch_page"] = bound_page
    return _collect_prospective_poll(**kwargs)


def test_first_seen_is_observed_response_time_not_provider_publication(
    tmp_path: Path,
) -> None:
    membership = _membership_authority(tmp_path)
    output = tmp_path / "poll"

    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=output,
        fetch_assets=_assets,
        fetch_page=lambda *_: _page(
            None, None, (_news(headline="Initial note"),)
        ),
        observed_at_utc=OBSERVED_AT,
        clock=_Clock(),
    )
    loaded = load_prospective_broker_action_poll(output)
    observation = loaded.observations.iloc[0]
    raw_page = _json_object(output / "raw_pages" / "batch-0000" / "page_000000.json")
    response_received = pd.Timestamp(raw_page["response_received_at_utc"])

    assert observation["revision_first_seen_at_utc"] == response_received
    assert response_received > pd.Timestamp(OBSERVED_AT)
    assert observation["revision_first_seen_at_utc"] != observation["published_at_utc"]
    assert observation["revision_first_seen_at_utc"] != observation["provider_updated_at_utc"]


def test_two_pages_preserve_distinct_revisions_and_content_hashes(
    tmp_path: Path,
) -> None:
    membership = _membership_authority(tmp_path)

    def fetch_page(
        symbols: str,
        start: datetime,
        end: datetime,
        token: str | None,
    ) -> AlpacaNewsPage:
        del symbols, start, end
        if token is None:
            return _page(
                None,
                "page-2",
                (
                    _news(
                        headline="Initial note",
                        content="Broker maintains rating.",
                        updated_at="2026-08-15T10:05:00Z",
                    ),
                ),
                received_seconds=2,
            )
        assert token == "page-2"
        return _page(
            "page-2",
            None,
            (
                _news(
                    headline="Corrected note",
                    content="Broker raises rating.",
                    updated_at="2026-08-15T10:10:00Z",
                ),
            ),
            received_seconds=3,
        )

    output = tmp_path / "poll"
    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=output,
        fetch_assets=_assets,
        fetch_page=fetch_page,
        observed_at_utc=OBSERVED_AT,
        clock=_Clock(),
    )

    observations = load_prospective_broker_action_poll(output).observations
    assert observations["provider_event_id"].tolist() == ["provider-event-1"] * 2
    assert observations["revision_id"].nunique() == 2
    assert observations["raw_sha256"].nunique() == 2
    assert observations["title"].tolist() == ["Initial note", "Corrected note"]
    assert observations["revision_first_seen_at_utc"].is_monotonic_increasing


def test_terminal_empty_page_publishes_known_zero_coverage(tmp_path: Path) -> None:
    membership = _membership_authority(tmp_path)
    output = tmp_path / "poll"

    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=output,
        fetch_assets=_assets,
        fetch_page=lambda *_: _page(None, None, ()),
        observed_at_utc=OBSERVED_AT,
        clock=_Clock(),
    )

    loaded = load_prospective_broker_action_poll(output)
    assert loaded.observations.empty
    assert loaded.source_collections["status"].tolist() == ["observed_empty"]
    assert loaded.source_collections["row_count"].tolist() == [0]
    assert loaded.source_collections["ticker"].tolist() == ["AAA"]


def test_stale_membership_identity_abstains_without_dropping_observation(
    tmp_path: Path,
) -> None:
    membership = _membership_authority(tmp_path, cutoff_date="2026-08-13")
    output = tmp_path / "poll"

    manifest = collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=output,
        fetch_assets=_assets,
        fetch_page=lambda *_: _page(None, None, (_news(),)),
        observed_at_utc=OBSERVED_AT,
        clock=_Clock(),
    )

    loaded = load_prospective_broker_action_poll(output)
    identity = loaded.identity_audit.iloc[0]
    observation = loaded.observations.iloc[0]
    assert not bool(identity["identity_eligible"])
    assert identity["identity_ineligible_reason"] == "membership_authority_stale"
    assert not bool(observation["identity_eligible"])
    assert observation["identity_ineligible_reason"] == "membership_authority_stale"
    assert manifest["event_observation_count"] == 1
    assert manifest["production_identity_event_count"] == 0


def test_previous_closed_new_york_membership_date_is_current_on_weekend(
    tmp_path: Path,
) -> None:
    weekend_observed_at = datetime(2026, 8, 16, 7, 0, tzinfo=UTC)
    membership = _membership_authority(tmp_path, cutoff_date="2026-08-15")
    output = tmp_path / "poll"

    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=output,
        fetch_assets=_assets,
        fetch_page=lambda *_: replace(
            _page(None, None, ()),
            retrieved_at_utc=weekend_observed_at + timedelta(seconds=2),
        ),
        observed_at_utc=weekend_observed_at,
        clock=_Clock(weekend_observed_at),
    )

    identity = load_prospective_broker_action_poll(output).identity_audit.iloc[0]
    assert bool(identity["identity_eligible"])
    assert identity["identity_ineligible_reason"] == ""


def test_previous_new_york_date_is_stale_on_weekday(tmp_path: Path) -> None:
    weekday_observed_at = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
    membership = _membership_authority(tmp_path, cutoff_date="2026-08-16")
    output = tmp_path / "poll"

    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=output,
        fetch_assets=_assets,
        fetch_page=lambda *_: replace(
            _page(None, None, ()),
            retrieved_at_utc=weekday_observed_at + timedelta(seconds=2),
        ),
        observed_at_utc=weekday_observed_at,
        clock=_Clock(weekday_observed_at),
    )

    identity = load_prospective_broker_action_poll(output).identity_audit.iloc[0]
    assert not bool(identity["identity_eligible"])
    assert identity["identity_ineligible_reason"] == "membership_authority_stale"


def test_resume_uses_archived_page_without_refetch(tmp_path: Path) -> None:
    membership = _membership_authority(tmp_path)
    output = tmp_path / "poll"
    clock = _Clock()
    first_calls: list[str | None] = []

    def interrupted_fetch(
        symbols: str,
        start: datetime,
        end: datetime,
        token: str | None,
    ) -> AlpacaNewsPage:
        del symbols, start, end
        first_calls.append(token)
        if token == "page-2":
            raise RuntimeError("temporary provider failure")
        return _page(
            None,
            "page-2",
            (_news(headline="Initial note"),),
        )

    incomplete = collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=output,
        fetch_assets=_assets,
        fetch_page=interrupted_fetch,
        observed_at_utc=OBSERVED_AT,
        clock=clock,
    )
    assert incomplete["status"] == "incomplete"
    assert first_calls == [None, "page-2"]

    resumed_calls: list[str | None] = []

    def resumed_fetch(
        symbols: str,
        start: datetime,
        end: datetime,
        token: str | None,
    ) -> AlpacaNewsPage:
        del symbols, start, end
        resumed_calls.append(token)
        return _page(
            token,
            None,
            (
                _news(
                    headline="Corrected note",
                    content="Broker raises rating.",
                    updated_at="2026-08-15T10:10:00Z",
                ),
                ),
            received_seconds=10,
        )

    complete = collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=output,
        fetch_assets=lambda: pytest.fail("resumed poll must reuse archived assets"),
        fetch_page=resumed_fetch,
        observed_at_utc=OBSERVED_AT,
        clock=clock,
    )

    assert complete["status"] == "complete"
    assert resumed_calls == ["page-2"]
    assert len(load_prospective_broker_action_poll(output).observations) == 2


def test_strict_replay_rejects_raw_page_tamper_and_extra_file(
    tmp_path: Path,
) -> None:
    membership = _membership_authority(tmp_path)
    original = tmp_path / "poll"
    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=original,
        fetch_assets=_assets,
        fetch_page=lambda *_: _page(None, None, (_news(),)),
        observed_at_utc=OBSERVED_AT,
        clock=_Clock(),
    )
    load_prospective_broker_action_poll(original)

    tampered = tmp_path / "tampered"
    shutil.copytree(original, tampered)
    page_path = tampered / "raw_pages" / "batch-0000" / "page_000000.json"
    page = _json_object(page_path)
    page["news"][0]["headline"] = "Tampered headline"
    _write_json(page_path, page)
    with pytest.raises(DataReadinessError):
        load_prospective_broker_action_poll(tampered)

    extra = tmp_path / "extra"
    shutil.copytree(original, extra)
    (extra / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="root inventory"):
        load_prospective_broker_action_poll(extra)


def test_generation_keeps_earliest_first_seen_and_all_content_revisions(
    tmp_path: Path,
) -> None:
    membership = _membership_authority(tmp_path)
    first = tmp_path / "poll-first"
    second = tmp_path / "poll-second"
    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=first,
        fetch_assets=_assets,
        fetch_page=lambda *_: _page(None, None, (_news(),)),
        observed_at_utc=OBSERVED_AT,
        clock=_Clock(),
    )
    first_seen = load_prospective_broker_action_poll(first).observations.iloc[0][
        "revision_first_seen_at_utc"
    ]
    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=second,
        fetch_assets=_assets,
        fetch_page=lambda *_: _page(
            None,
            None,
            (
                _news(),
                _news(
                    headline="Corrected broker action",
                    content="Broker raises rating.",
                ),
            ),
            received_seconds=62,
        ),
        observed_at_utc=OBSERVED_AT + timedelta(minutes=1),
        previous_poll_directory=first,
        clock=_Clock(OBSERVED_AT + timedelta(minutes=1)),
    )
    output = tmp_path / "generation"
    manifest = publish_prospective_broker_action_generation(
        poll_directories=[second, first],
        output_directory=output,
    )

    generation = load_prospective_broker_action_generation(output)
    revisions = generation.revisions.sort_values("title", kind="stable")
    assert manifest["revision_count"] == 2
    assert revisions["revision_id"].nunique() == 2
    original = revisions[revisions["title"].eq("Broker action")].iloc[0]
    assert original["observation_count"] == 2
    assert original["revision_first_seen_at_utc"] == first_seen
    assert bool(revisions["provider_timestamp_anomaly"].all())
    assert not bool(manifest["training_eligible"])
    assert not bool(manifest["serving_eligible"])


def test_generation_replay_rejects_parent_poll_tamper(tmp_path: Path) -> None:
    membership = _membership_authority(tmp_path)
    poll = tmp_path / "poll"
    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=poll,
        fetch_assets=_assets,
        fetch_page=lambda *_: _page(None, None, (_news(),)),
        observed_at_utc=OBSERVED_AT,
        clock=_Clock(),
    )
    generation = tmp_path / "generation"
    publish_prospective_broker_action_generation(
        poll_directories=[poll],
        output_directory=generation,
    )
    page_path = poll / "raw_pages" / "batch-0000" / "page_000000.json"
    page = _json_object(page_path)
    page["news"][0]["headline"] = "Tampered after generation publication"
    _write_json(page_path, page)

    with pytest.raises(DataReadinessError, match="raw inventory"):
        load_prospective_broker_action_generation(generation)


def test_poll_replay_rejects_authority_and_membership_parent_tamper(
    tmp_path: Path,
) -> None:
    membership = _membership_authority(tmp_path)
    poll = tmp_path / "poll"
    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=poll,
        fetch_assets=_assets,
        fetch_page=lambda *_: _page(None, None, (_news(),)),
        observed_at_utc=OBSERVED_AT,
        clock=_Clock(),
    )
    authority_path = poll / "_authority.json"
    authority = _json_object(authority_path)
    authority["production_ready"] = True
    _write_json(authority_path, authority)
    with pytest.raises(DataReadinessError, match="manifest or authority"):
        load_prospective_broker_action_poll(poll)

    authority["production_ready"] = False
    _write_json(authority_path, authority)
    membership_manifest_path = membership / "_manifest.json"
    membership_manifest = _json_object(membership_manifest_path)
    membership_manifest["cutoff_date"] = "2026-08-14"
    _write_json(membership_manifest_path, membership_manifest)
    with pytest.raises(DataReadinessError, match="membership authority"):
        load_prospective_broker_action_poll(poll)


@pytest.mark.parametrize(
    ("available_at", "duplicate_interval", "reason"),
    [
        (
            "2026-08-16T00:00:00Z",
            False,
            "membership_identity_not_available_at_poll",
        ),
        (
            "2020-01-01T00:00:00Z",
            True,
            "multiple_active_membership_intervals",
        ),
    ],
)
def test_membership_availability_and_interval_cardinality_abstain(
    tmp_path: Path,
    available_at: str,
    duplicate_interval: bool,
    reason: str,
) -> None:
    membership = _membership_authority(
        tmp_path,
        available_at=available_at,
        duplicate_interval=duplicate_interval,
    )
    output = tmp_path / "poll"
    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=output,
        fetch_assets=_assets,
        fetch_page=lambda *_: _page(None, None, (_news(),)),
        observed_at_utc=OBSERVED_AT,
        clock=_Clock(),
    )
    identity = load_prospective_broker_action_poll(output).identity_audit.iloc[0]
    assert not bool(identity["identity_eligible"])
    assert identity["identity_ineligible_reason"] == reason


def test_asset_identity_change_abstains_until_transition_is_resolved(
    tmp_path: Path,
) -> None:
    membership = _membership_authority(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=first,
        fetch_assets=_assets,
        fetch_page=lambda *_: _page(None, None, (_news(),)),
        observed_at_utc=OBSERVED_AT,
        clock=_Clock(),
    )
    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=second,
        fetch_assets=lambda: _assets_with_id("alpaca-asset-replacement"),
        fetch_page=lambda *_: _page(
            None,
            None,
            (_news(headline="New revision"),),
            received_seconds=62,
        ),
        observed_at_utc=OBSERVED_AT + timedelta(minutes=1),
        previous_poll_directory=first,
        clock=_Clock(OBSERVED_AT + timedelta(minutes=1)),
    )
    identity = load_prospective_broker_action_poll(second).identity_audit.iloc[0]
    assert not bool(identity["identity_eligible"])
    assert (
        identity["identity_ineligible_reason"]
        == "unresolved_identity_change_from_previous_poll"
    )

    third = tmp_path / "third"
    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=third,
        fetch_assets=lambda: _assets_with_id("alpaca-asset-replacement"),
        fetch_page=lambda *_: _page(
            None,
            None,
            (_news(headline="Later revision"),),
            received_seconds=122,
        ),
        observed_at_utc=OBSERVED_AT + timedelta(minutes=2),
        previous_poll_directory=second,
        clock=_Clock(OBSERVED_AT + timedelta(minutes=2)),
    )
    third_identity = load_prospective_broker_action_poll(
        third
    ).identity_audit.iloc[0]
    assert not bool(third_identity["identity_eligible"])
    assert bool(third_identity["identity_quarantined"])
    assert (
        third_identity["identity_ineligible_reason"]
        == "unresolved_identity_change_from_previous_poll"
    )


def test_asset_failure_is_recorded_before_reraise(tmp_path: Path) -> None:
    membership = _membership_authority(tmp_path)
    output = tmp_path / "poll"

    with pytest.raises(RuntimeError, match="asset outage"):
        collect_prospective_broker_action_poll(
            membership_authority_directory=membership,
            output_directory=output,
            fetch_assets=lambda: (_ for _ in ()).throw(
                RuntimeError("asset outage")
            ),
            fetch_page=lambda *_: pytest.fail("news must not be fetched"),
            observed_at_utc=OBSERVED_AT,
            clock=_Clock(),
        )

    attempts = list((output / "attempts" / "asset-snapshot").glob("*.json"))
    assert len(attempts) == 1
    assert _json_object(attempts[0])["error"] == "RuntimeError: asset outage"
    assert _json_object(output / "_status.json")["failed_stage"] == "asset_snapshot"


def test_malformed_asset_response_is_recorded_before_reraise(
    tmp_path: Path,
) -> None:
    membership = _membership_authority(tmp_path)
    output = tmp_path / "poll"

    with pytest.raises(DataReadinessError, match="final URL"):
        _collect_prospective_poll(
            membership_authority_directory=membership,
            intraday_bar_dataset_directory=_a43_dataset(membership),
            registry_directory=tmp_path / "registry",
            output_directory=output,
            fetch_assets=lambda: replace(_assets(), final_url=None),
            fetch_page=lambda *_: pytest.fail("news must not be fetched"),
            observed_at_utc=OBSERVED_AT,
            clock=_Clock(),
        )

    attempts = list((output / "attempts" / "asset-snapshot").glob("*.json"))
    assert len(attempts) == 1
    assert "final URL evidence" in _json_object(attempts[0])["error"]
    assert _json_object(output / "_status.json")["failed_stage"] == "asset_snapshot"


def test_poll_rejects_membership_not_bound_to_a43_before_fetch(
    tmp_path: Path,
) -> None:
    first = _membership_authority(tmp_path, cutoff_date="2026-08-15")
    a43 = _a43_dataset(first)
    second = _membership_authority(tmp_path, cutoff_date="2026-08-14")

    with pytest.raises(DataReadinessError, match="A4.3 namespace"):
        _collect_prospective_poll(
            membership_authority_directory=second,
            intraday_bar_dataset_directory=a43,
            registry_directory=tmp_path / "registry",
            output_directory=tmp_path / "poll",
            fetch_assets=lambda: pytest.fail("identity mismatch must precede fetch"),
            fetch_page=lambda *_: pytest.fail("identity mismatch must precede fetch"),
            observed_at_utc=OBSERVED_AT,
            clock=_Clock(),
        )


def test_poll_rejects_wrong_news_query_before_known_zero(
    tmp_path: Path,
) -> None:
    membership = _membership_authority(tmp_path)
    a43 = _a43_dataset(membership)
    asset_url = (
        "https://api.alpaca.markets/v2/assets?"
        + urlencode({"status": "active", "asset_class": "us_equity"})
    )
    bad_url = (
        "https://data.alpaca.markets/v1beta1/news?"
        + urlencode(
            {
                "symbols": "ZZZ",
                "start": (OBSERVED_AT - timedelta(hours=25)).isoformat(),
                "end": OBSERVED_AT.isoformat(),
                "sort": "asc",
                "limit": "50",
                "include_content": "true",
            }
        )
    )

    result = _collect_prospective_poll(
        membership_authority_directory=membership,
        intraday_bar_dataset_directory=a43,
        registry_directory=tmp_path / "registry",
        output_directory=tmp_path / "poll",
        fetch_assets=lambda: replace(
            _assets(),
            requested_url=asset_url,
            final_url=asset_url,
            redirect_chain=(),
        ),
        fetch_page=lambda *_: replace(
            _page(None, None, ()),
            requested_url=bad_url,
            final_url=bad_url,
            redirect_chain=(),
        ),
        observed_at_utc=OBSERVED_AT,
        clock=_Clock(),
    )

    assert result["status"] == "incomplete"
    assert not (tmp_path / "poll" / "_authority.json").exists()


def test_current_membership_extension_preserves_a43_namespace(
    tmp_path: Path,
) -> None:
    base = _membership_authority(tmp_path, cutoff_date="2026-08-14")
    current = _membership_authority(tmp_path, cutoff_date="2026-08-15")
    output = tmp_path / "poll"

    collect_prospective_broker_action_poll(
        membership_authority_directory=current,
        intraday_bar_dataset_directory=_a43_dataset(base),
        output_directory=output,
        fetch_assets=_assets,
        fetch_page=lambda *_: _page(None, None, (_news(),)),
        observed_at_utc=OBSERVED_AT,
        clock=_Clock(),
    )

    identity = load_prospective_broker_action_poll(output).identity_audit.iloc[0]
    assert bool(identity["identity_eligible"])


def test_asset_response_before_cutoff_is_recorded_and_rejected(
    tmp_path: Path,
) -> None:
    membership = _membership_authority(tmp_path)
    output = tmp_path / "poll"
    asset_url = (
        "https://api.alpaca.markets/v2/assets?"
        + urlencode({"status": "active", "asset_class": "us_equity"})
    )

    with pytest.raises(DataReadinessError, match="continuous-coverage"):
        _collect_prospective_poll(
            membership_authority_directory=membership,
            intraday_bar_dataset_directory=_a43_dataset(membership),
            registry_directory=tmp_path / "registry",
            output_directory=output,
            fetch_assets=lambda: replace(
                _assets(),
                requested_url=asset_url,
                final_url=asset_url,
                redirect_chain=(),
                retrieved_at_utc=OBSERVED_AT - timedelta(seconds=1),
            ),
            fetch_page=lambda *_: pytest.fail("news must not be fetched"),
            observed_at_utc=OBSERVED_AT,
            clock=_Clock(),
        )

    assert list((output / "attempts" / "asset-snapshot").glob("*.json"))


def test_path_overlap_fails_before_output_creation(tmp_path: Path) -> None:
    membership = _membership_authority(tmp_path)
    output = tmp_path / "poll"

    with pytest.raises(DataReadinessError, match="paths overlap"):
        _collect_prospective_poll(
            membership_authority_directory=membership,
            intraday_bar_dataset_directory=_a43_dataset(membership),
            registry_directory=membership / "registry",
            output_directory=output,
            fetch_assets=lambda: pytest.fail("path validation must precede fetch"),
            fetch_page=lambda *_: pytest.fail("path validation must precede fetch"),
            observed_at_utc=OBSERVED_AT,
            clock=_Clock(),
        )

    assert not output.exists()


def test_resume_reconstructs_assets_after_raw_only_crash_boundary(
    tmp_path: Path,
) -> None:
    membership = _membership_authority(tmp_path)
    output = tmp_path / "poll"
    incomplete = collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=output,
        fetch_assets=_assets,
        fetch_page=lambda *_: (_ for _ in ()).throw(RuntimeError("stop")),
        observed_at_utc=OBSERVED_AT,
        clock=_Clock(),
    )
    assert incomplete["status"] == "incomplete"
    for path in (
        output / "assets.parquet",
        output / "assets.parquet.manifest.json",
        output / "assets.parquet.lock",
    ):
        path.unlink()
    complete = collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=output,
        fetch_assets=lambda: pytest.fail("raw asset body must be replayed"),
        fetch_page=lambda *_: _page(None, None, (_news(),), received_seconds=10),
        observed_at_utc=OBSERVED_AT,
        clock=_Clock(OBSERVED_AT + timedelta(seconds=5)),
    )
    assert complete["status"] == "complete"


def test_registry_rejects_duplicate_scheduled_cutoffs_before_fetch(
    tmp_path: Path,
) -> None:
    membership = _membership_authority(tmp_path)
    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=tmp_path / "first",
        fetch_assets=_assets,
        fetch_page=lambda *_: _page(None, None, (_news(),)),
        observed_at_utc=OBSERVED_AT,
        clock=_Clock(),
    )
    with pytest.raises(DataReadinessError, match="already claimed"):
        collect_prospective_broker_action_poll(
            membership_authority_directory=membership,
            output_directory=tmp_path / "second",
            fetch_assets=lambda: pytest.fail("duplicate cutoff must fail before fetch"),
            fetch_page=lambda *_: pytest.fail("duplicate cutoff must fail before fetch"),
            observed_at_utc=OBSERVED_AT,
            clock=_Clock(),
        )


def test_previous_poll_must_precede_child_cutoff(tmp_path: Path) -> None:
    membership = _membership_authority(tmp_path)
    first = tmp_path / "first"
    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=first,
        fetch_assets=_assets,
        fetch_page=lambda *_: _page(None, None, ()),
        observed_at_utc=OBSERVED_AT,
        clock=_Clock(),
    )

    with pytest.raises(DataReadinessError, match="must precede"):
        collect_prospective_broker_action_poll(
            membership_authority_directory=membership,
            output_directory=tmp_path / "child",
            fetch_assets=lambda: pytest.fail("temporal failure must precede fetch"),
            fetch_page=lambda *_: pytest.fail("temporal failure must precede fetch"),
            observed_at_utc=OBSERVED_AT - timedelta(minutes=1),
            previous_poll_directory=first,
            clock=_Clock(),
        )


def test_registry_commit_tamper_fails_strict_replay(tmp_path: Path) -> None:
    membership = _membership_authority(tmp_path)
    output = tmp_path / "poll"
    collect_prospective_broker_action_poll(
        membership_authority_directory=membership,
        output_directory=output,
        fetch_assets=_assets,
        fetch_page=lambda *_: _page(None, None, ()),
        observed_at_utc=OBSERVED_AT,
        clock=_Clock(),
    )
    commit = next((tmp_path / "poll-registry" / "commits").rglob("*.json"))
    payload = _json_object(commit)
    payload["raw_pages_inventory_sha256"] = "0" * 64
    _write_json(commit, payload)

    with pytest.raises(DataReadinessError, match="registry"):
        load_prospective_broker_action_poll(output)


def _membership_authority(
    tmp_path: Path,
    *,
    cutoff_date: str = "2026-08-15",
    available_at: str = "2020-01-01T00:00:00Z",
    duplicate_interval: bool = False,
) -> Path:
    root = tmp_path / f"memberships-{cutoff_date}"
    root.mkdir()
    parent_lineage = {
        "anchor_file_sha256": "1" * 64,
        "anchor_semantic_sha256": "2" * 64,
        "event_authority_sha256": "3" * 64,
        "event_set_sha256": "4" * 64,
        "raw_authority_sha256": "5" * 64,
        "raw_manifest_sha256": "6" * 64,
        "transition_authority_sha256": "7" * 64,
        "transition_set_sha256": "8" * 64,
    }
    request_payload: dict[str, Any] = {
        "schema": "edge_rebuild.sp500_membership_request.v1",
        "reconstruction_schema": "edge_rebuild.sp500_membership_reconstruction.v1",
        "start_date": "2020-01-01",
        "cutoff_date": cutoff_date,
        "parent_lineage": parent_lineage,
    }
    request_sha256 = json_sha256(request_payload)
    _write_json(
        root / "_request.json",
        {**request_payload, "request_sha256": request_sha256},
    )

    membership_rows = [
        {
                "ticker": "AAA",
                "security_id": "security:aaa",
                "effective_from_utc": pd.Timestamp("2020-01-01T00:00:00Z"),
                "effective_to_utc": pd.NaT,
                "available_at_utc": pd.Timestamp(available_at),
        }
    ]
    if duplicate_interval:
        membership_rows.append(dict(membership_rows[0]))
    memberships = pd.DataFrame(membership_rows)
    membership_path = root / "memberships.parquet"
    write_canonical_artifact(
        memberships,
        membership_path,
        artifact_type="memberships",
        audit=_passing_audit("membership_fixture", len(memberships)),
        inputs={"request_sha256": request_sha256, **parent_lineage},
        production_ready=False,
    )
    manifest = {
        "schema": "edge_rebuild.sp500_membership_manifest.v1",
        "status": "complete",
        "request_sha256": request_sha256,
        "cutoff_date": cutoff_date,
        "universe_sha256": json_sha256(["security:aaa", "AAA"]),
        "parent_lineage": parent_lineage,
        "membership_manifest_sha256": file_sha256(
            membership_path.with_suffix(".parquet.manifest.json")
        ),
        "membership_artifact": {
            "path": membership_path.name,
            "sha256": file_sha256(membership_path),
            "bytes": membership_path.stat().st_size,
        },
    }
    manifest_path = root / "_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        root / "_authority.json",
        {
            "schema": "edge_rebuild.sp500_membership_authority.v1",
            "state": "membership_complete",
            "artifact": manifest_path.name,
            "artifact_sha256": file_sha256(manifest_path),
            "request_sha256": request_sha256,
            "parent_lineage": parent_lineage,
        },
    )
    return root


def _a43_dataset(membership: Path) -> Path:
    root = membership.parent / f"a43-{membership.name}"
    if root.exists():
        return root
    root.mkdir()
    membership_request = _json_object(membership / "_request.json")
    membership_manifest = _json_object(membership / "_manifest.json")
    membership_record = membership_manifest["membership_artifact"]
    assert isinstance(membership_record, dict)
    parent_lineage = {
        "membership_authority_sha256": file_sha256(
            membership / "_authority.json"
        ),
        "membership_manifest_sha256": file_sha256(
            membership / "_manifest.json"
        ),
        "membership_table_sha256": membership_record["sha256"],
    }
    transformation = _transformation_identity()
    request_payload = {
        "schema": "edge_rebuild.intraday_bar_dataset.v1",
        "membership_authority_directory": str(membership.resolve()),
        "parent_lineage": parent_lineage,
        "parent_lineage_sha256": json_sha256(parent_lineage),
        "planned_sessions": ["2026-08-14"],
        "membership_request_sha256": membership_request["request_sha256"],
        "transformation": transformation,
        "transformation_sha256": transformation["sha256"],
    }
    request_sha256 = json_sha256(request_payload)
    _write_json(
        root / "_request.json",
        {**request_payload, "request_sha256": request_sha256},
    )
    unit_root = root / "sessions" / "session_date_et=2026-08-14"
    unit_root.mkdir(parents=True)
    rows_path = unit_root / "rows.parquet"
    pd.DataFrame(
        {
            "session_date_et": pd.Series(dtype="string"),
            "ticker": pd.Series(dtype="string"),
            "decision_time_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "dataset_eligible": pd.Series(dtype="bool"),
        }
    ).to_parquet(rows_path, index=False)
    _write_json(unit_root / "audit.json", {"status": "pass"})
    parquet = pq.ParquetFile(rows_path)
    parquet_schema = _arrow_schema_record(parquet.schema_arrow)
    session_request_sha256 = json_sha256(
        {"session_date_et": "2026-08-14", "request_sha256": request_sha256}
    )
    unit = {
        "schema": "edge_rebuild.intraday_bar_dataset_session_unit.v1",
        "state": "complete",
        "session_date_et": "2026-08-14",
        "request_sha256": request_sha256,
        "transformation_sha256": transformation["sha256"],
        "session_request_sha256": session_request_sha256,
        "rows": 0,
        "dataset_eligible_rows": 0,
        "ticker_count": 0,
        "rows_sha256": file_sha256(rows_path),
        "audit_sha256": file_sha256(unit_root / "audit.json"),
        "parquet_schema": parquet_schema,
        "parquet_schema_sha256": json_sha256(parquet_schema),
    }
    _write_json(unit_root / "_unit.json", unit)
    units = [unit]
    manifest = {
        "schema": "edge_rebuild.intraday_bar_dataset.v1",
        "state": "complete",
        "request_sha256": request_sha256,
        "parent_lineage": parent_lineage,
        "parent_lineage_sha256": json_sha256(parent_lineage),
        "session_units": units,
        "session_unit_inventory_sha256": json_sha256(units),
        "summary": {
            "completed_sessions": 1,
            "rows": 0,
            "dataset_eligible_rows": 0,
        },
    }
    _write_json(root / "_manifest.json", manifest)
    _write_json(
        root / "_authority.json",
        {
            "schema": "edge_rebuild.intraday_bar_dataset_authority.v1",
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(root / "_manifest.json"),
            "request_sha256": request_sha256,
            "session_unit_inventory_sha256": json_sha256(units),
            "sessions": 1,
            "rows": 0,
        },
    )
    return root


def _assets() -> AlpacaAssetSnapshot:
    return _assets_with_id("alpaca-asset-aaa")


def _assets_with_id(asset_id: str) -> AlpacaAssetSnapshot:
    rows = [
        {
            "id": asset_id,
            "symbol": "AAA",
            "status": "active",
            "exchange": "NYSE",
            "tradable": True,
        }
    ]
    body = json.dumps(rows, sort_keys=True).encode()
    return AlpacaAssetSnapshot(
        assets=pd.DataFrame(rows),
        raw_body=body,
        response_headers={"content-type": "application/json"},
        requested_url="https://api.alpaca.markets/v2/assets",
        status_code=200,
        retrieved_at_utc=OBSERVED_AT + timedelta(seconds=1),
    )


def _page(
    request_token: str | None,
    next_token: str | None,
    news: tuple[dict[str, object], ...],
    *,
    received_seconds: int = 2,
) -> AlpacaNewsPage:
    payload = {"news": list(news), "next_page_token": next_token}
    body = json.dumps(payload, sort_keys=True).encode()
    return AlpacaNewsPage(
        request_page_token=request_token,
        next_page_token=next_token,
        news=news,
        response_headers={"content-type": "application/json"},
        raw_payload=payload,
        raw_body=body,
        requested_url="https://data.alpaca.markets/v1beta1/news",
        status_code=200,
        retrieved_at_utc=OBSERVED_AT + timedelta(seconds=received_seconds),
    )


def _news(
    *,
    headline: str = "Broker action",
    content: str = "Broker maintains rating.",
    updated_at: str = "2026-08-15T10:05:00Z",
) -> dict[str, object]:
    return {
        "id": "provider-event-1",
        "created_at": "2026-08-15T10:00:00Z",
        "updated_at": updated_at,
        "headline": headline,
        "source": "benzinga",
        "symbols": ["AAA"],
        "url": "https://example.test/news/provider-event-1",
        "summary": content,
        "content": content,
    }


def _passing_audit(name: str, rows: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name=name,
                status="pass",
                failures=0,
                rows_checked=rows,
                detail="synthetic authority fixture",
            ),
        )
    )


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
