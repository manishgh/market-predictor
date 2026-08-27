from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
import requests

from market_predictor.canonical.store import load_canonical_artifact
from market_predictor.catalysts.global_events.collection import (
    collect_live_gdelt_global_events,
    load_gdelt_global_event_collection,
)
from market_predictor.catalysts.global_events.decision_authority import (
    publish_global_event_authority,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.sources.gdelt import (
    GDELT_DOCUMENT_ENDPOINT,
    GdeltDocumentRequest,
    GdeltDocumentResult,
    fetch_gdelt_documents,
    validate_gdelt_document_request,
)

START = datetime(2025, 1, 7, 19, 0, tzinfo=UTC)
END = datetime(2025, 1, 10, 20, 0, tzinfo=UTC)
FETCHED = datetime(2025, 1, 10, 20, 1, tzinfo=UTC)
SCORED = datetime(2025, 1, 10, 20, 2, tzinfo=UTC)
COMPLETED = datetime(2025, 1, 10, 20, 3, tzinfo=UTC)
QUERIES = (
    "(war OR sanctions OR shipping OR energy)",
    "(semiconductor OR export controls)",
)


class _Clock:
    def __init__(self, values: Sequence[datetime]) -> None:
        self._values = iter(values)

    def __call__(self) -> datetime:
        return next(self._values)


class _Scorer:
    model_name = "fixture-finbert"
    model_revision = "fixture-v1"

    def __init__(self) -> None:
        self.calls = 0

    def score_texts(self, texts: list[str], batch_size: int = 16) -> pd.DataFrame:
        self.calls += 1
        return pd.DataFrame(
            {
                "sentiment_numeric": [0.6 - index * 0.1 for index in range(len(texts))],
                "relevance": [0.9 for _ in texts],
            }
        )


def _fetch_result(
    records: Sequence[Mapping[str, object]],
    *,
    complete: bool = True,
    errors: tuple[str, ...] = (),
) -> GdeltDocumentResult:
    tagged_records = [
        {**dict(record), "collection_query": record.get("collection_query", QUERIES[0])}
        for record in records
    ]
    return GdeltDocumentResult(
        records=tagged_records,
        complete=complete,
        completed_queries=QUERIES if complete else (),
        errors=errors,
    )


def test_live_collection_publishes_observed_causal_artifacts_consumable_by_authority(
    tmp_path: Path,
) -> None:
    scorer = _Scorer()
    raw = _article(
        title="Shipping disruption raises energy risk",
        url="https://example.com/story#fragment",
        published_at="2025-01-10T19:30:00Z",
        first_seen_at_utc="2025-01-10T19:30:00Z",
        available_at_utc="2025-01-10T19:30:00Z",
        availability_policy="provider_publication_proxy",
    )
    collection = collect_live_gdelt_global_events(
        _request(),
        tmp_path / "gdelt",
        scorer=scorer,
        fetch=lambda _: _fetch_result([raw]),
        clock=_Clock([END, END, END, END]),
    )

    assert scorer.calls == 1
    row = collection.events.iloc[0]
    assert row["ticker"] == "MARKET"
    assert row["security_id"] == "market:global"
    assert row["source_family"] == "gdelt"
    assert row["availability_policy"] == "observed"
    assert pd.Timestamp(row["first_seen_at_utc"]) == pd.Timestamp(END)
    assert pd.Timestamp(row["available_at_utc"]) == pd.Timestamp(END)
    assert pd.Timestamp(row["feature_available_at_utc"]) == pd.Timestamp(END)
    assert pd.Timestamp(row["first_seen_at_utc"]) != pd.Timestamp(row["published_at_utc"])
    assert len(str(row["raw_sha256"])) == 64
    event_inputs = load_canonical_artifact(collection.events_path, expected_type="events")[1]["inputs"]
    coverage_inputs = load_canonical_artifact(
        collection.source_collections_path,
        expected_type="source_collections",
    )[1]["inputs"]
    request_payload = collection.manifest["request"]
    assert isinstance(event_inputs, Mapping)
    assert isinstance(coverage_inputs, Mapping)
    assert isinstance(request_payload, Mapping)
    assert event_inputs["collection_request_sha256"] == coverage_inputs["collection_request_sha256"]
    assert event_inputs["source_policy_sha256"] == request_payload["source_policy_sha256"]
    assert event_inputs["source_policy_sha256"] != request_payload["query_policy_sha256"]

    authority = publish_global_event_authority(
        pd.DataFrame({"decision_time_utc": [END]}),
        [collection.events_path],
        [collection.source_collections_path],
        tmp_path / "authority",
        required_historical_sources=("gdelt",),
        production_ready=True,
    )
    assert authority.decisions.loc[0, "global_event_count_1d"] == 1.0


def test_empty_success_publishes_observed_empty_without_calling_scorer(tmp_path: Path) -> None:
    scorer = _Scorer()
    collection = collect_live_gdelt_global_events(
        _request(),
        tmp_path / "empty",
        scorer=scorer,
        fetch=lambda _: _fetch_result([]),
        clock=_Clock([END, FETCHED, COMPLETED]),
    )

    assert collection.events.empty
    assert scorer.calls == 0
    coverage = collection.source_collections.iloc[0]
    assert coverage["status"] == "observed_empty"
    assert coverage["row_count"] == 0


def test_provider_proxy_and_backdated_availability_are_never_trusted(tmp_path: Path) -> None:
    publication = "2025-01-10T19:30:00Z"
    collection = collect_live_gdelt_global_events(
        _request(),
        tmp_path / "observed-only",
        scorer=_Scorer(),
        fetch=lambda _: _fetch_result(
            records=[
                _article(
                    title="Provider attempts to supply proxy availability",
                    url="https://example.com/proxy",
                    published_at=publication,
                    first_seen_at_utc=publication,
                    available_at_utc=publication,
                    feature_available_at_utc=publication,
                    availability_policy="provider_publication_proxy",
                )
            ],
        ),
        clock=_Clock([END, FETCHED, SCORED, COMPLETED]),
    )

    event = collection.events.iloc[0]
    assert event["availability_policy"] == "observed"
    assert pd.Timestamp(event["first_seen_at_utc"]) == pd.Timestamp(FETCHED)
    assert pd.Timestamp(event["available_at_utc"]) == pd.Timestamp(FETCHED)
    assert pd.Timestamp(event["feature_available_at_utc"]) == pd.Timestamp(SCORED)


def test_duplicates_are_deterministic_and_raw_hash_is_preserved(tmp_path: Path) -> None:
    first = _article(title="B", url="https://example.com/duplicate", published_at="2025-01-10T19:30:00Z")
    second = _article(title="A", url="https://example.com/duplicate", published_at="2025-01-10T19:30:00Z")
    collection = collect_live_gdelt_global_events(
        _request(),
        tmp_path / "deduplicated",
        scorer=_Scorer(),
        fetch=lambda _: _fetch_result([first, second, first]),
        clock=_Clock([END, FETCHED, SCORED, COMPLETED]),
    )
    reversed_collection = collect_live_gdelt_global_events(
        _request(),
        tmp_path / "deduplicated-reversed",
        scorer=_Scorer(),
        fetch=lambda _: _fetch_result([second, first, first]),
        clock=_Clock([END, FETCHED, SCORED, COMPLETED]),
    )

    assert len(collection.events) == 1
    assert collection.manifest["duplicate_rows"] == 2
    columns = ["event_id", "title", "raw_sha256"]
    pd.testing.assert_frame_equal(collection.events[columns], reversed_collection.events[columns])


def test_future_provider_publication_fails_closed_without_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "future"
    with pytest.raises(DataReadinessError, match="future"):
        collect_live_gdelt_global_events(
            _request(),
            output,
            scorer=_Scorer(),
            fetch=lambda _: _fetch_result(
                records=[
                    _article(
                        title="Impossible future article",
                        url="https://example.com/future",
                        published_at="2025-01-10T20:02:00Z",
                    )
                ],
            ),
            clock=_Clock([END, FETCHED]),
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".future.*.tmp"))


@pytest.mark.parametrize(
    "fetch_result,match",
    [
        (_fetch_result([], complete=False), "partial|truncated"),
        (_fetch_result([], errors=("timeout on shard 2",)), "reported errors"),
        (
            GdeltDocumentResult(
                records=[],
                complete=True,
                completed_queries=QUERIES[:1],
            ),
            "exact immutable query policy",
        ),
    ],
)
def test_partial_or_error_response_fails_closed(
    tmp_path: Path,
    fetch_result: GdeltDocumentResult,
    match: str,
) -> None:
    output = tmp_path / match.replace("|", "-")
    with pytest.raises(DataReadinessError, match=match):
        collect_live_gdelt_global_events(
            _request(),
            output,
            scorer=_Scorer(),
            fetch=lambda _: fetch_result,
            clock=_Clock([END, FETCHED]),
        )
    assert not output.exists()


def test_loader_rejects_request_replay_and_hash_tamper(tmp_path: Path) -> None:
    output = tmp_path / "collection"
    collection = collect_live_gdelt_global_events(
        _request(),
        output,
        scorer=_Scorer(),
        fetch=lambda _: _fetch_result(
            records=[_article(title="Event", url="https://example.com/event", published_at="2025-01-10T19:30:00Z")],
        ),
        clock=_Clock([END, FETCHED, SCORED, COMPLETED]),
    )
    with pytest.raises(DataReadinessError, match="another request"):
        load_gdelt_global_event_collection(
            output,
            expected_collection_request_sha256="0" * 64,
        )
    with pytest.raises(DataReadinessError, match="immutable"):
        collect_live_gdelt_global_events(
            _request(),
            output,
            scorer=_Scorer(),
            fetch=lambda _: _fetch_result([]),
        )

    manifest_path = output / "events.parquet.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"]["collection_request_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataReadinessError, match="artifact record|lineage"):
        load_gdelt_global_event_collection(collection.directory)


def test_collection_preserves_frozen_transport_and_lineage_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    response_bodies = [
        (
            b'{"articles":[{"title":"Shipping disruption raises energy risk",'
            b'"url":"https://example.com/story#fragment",'
            b'"seendate":"2025-01-10T19:30:00Z",'
            b'"domain":"example.com","summary":"Global macro event"}]}'
        ),
        b'{"articles":[]}',
    ]
    responses = [_http_response(200, raw=body) for body in response_bodies]
    calls: list[tuple[str, Mapping[str, object], float]] = []

    def fake_get(
        url: str,
        *,
        params: Mapping[str, object],
        timeout: float,
        allow_redirects: bool,
    ) -> requests.Response:
        assert allow_redirects is False
        calls.append((url, params, timeout))
        return responses.pop(0)

    monkeypatch.setattr(requests, "get", fake_get)
    request = _request()
    normalized = validate_gdelt_document_request(request)
    fetched = fetch_gdelt_documents(request)
    collection = collect_live_gdelt_global_events(
        request,
        tmp_path / "lineage-characterization",
        scorer=_Scorer(),
        fetch=lambda _: fetched,
        clock=_Clock([END, FETCHED, SCORED, COMPLETED]),
        scorer_identity="fixture-finbert|revision=fixture-v1",
    )

    assert normalized == GdeltDocumentRequest(
        queries=QUERIES,
        requested_start_utc=START,
        requested_end_utc=END,
        max_records=100,
        timeout_seconds=30.0,
    )
    assert calls == [
        (
            GDELT_DOCUMENT_ENDPOINT,
            {
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": 100,
                "startdatetime": "20250107190000",
                "enddatetime": "20250110200000",
            },
            30.0,
        )
        for query in QUERIES
    ]
    assert fetched.raw_response_sha256 == (
        "fffda4358a7061552d96d59b62d5c90684cef4863e3c718dd53527e4d7efb9da"
    )
    assert collection.manifest["request"] == {
        "schema": "edge_rebuild.gdelt_global_collection_request.v2",
        "source_family": "gdelt",
        "global_identity": {"ticker": "MARKET", "security_id": "market:global"},
        "queries": list(QUERIES),
        "query_policy_sha256": "6d1f4263c3d869e6345dcb244b732181695d575873fced93e3ad11b22e35dc35",
        "source_policy_sha256": "01988ee89958ce6fc26e71fcf36c379054eade52d920787f8acf37e05766eb2c",
        "requested_start_utc": "2025-01-07T19:00:00+00:00",
        "requested_end_utc": "2025-01-10T20:00:00+00:00",
        "max_records": 100,
        "endpoint": "https://api.gdeltproject.org/api/v2/doc/doc",
        "availability_policy": "observed collection and scoring timestamps only",
        "scorer_identity": "fixture-finbert|revision=fixture-v1",
        "scorer_batch_size": 16,
    }
    assert collection.manifest["collection_request_sha256"] == (
        "ebcd130dfa29d148357f2e6dcd6980402b7412f753b5f38695ee4f3364e4b674"
    )
    assert collection.manifest["raw_response_sha256"] == fetched.raw_response_sha256
    assert collection.events.loc[0, "raw_sha256"] == (
        "42c25d6efda36dd8eb8307aa207510c443f1bed1c10f0fa8ef1283089f267411"
    )


def test_query_families_remain_distinguishable_after_canonicalization(tmp_path: Path) -> None:
    shared = _article(
        title="Shared flashpoint article",
        url="https://example.com/shared",
        published_at="2025-01-10T19:30:00Z",
    )
    collection = collect_live_gdelt_global_events(
        _request(),
        tmp_path / "query-families",
        scorer=_Scorer(),
        fetch=lambda _: _fetch_result(
            [
                {**shared, "collection_query": QUERIES[0]},
                {**shared, "collection_query": QUERIES[1]},
            ]
        ),
        clock=_Clock([END, FETCHED, SCORED, COMPLETED]),
    )

    assert len(collection.events) == 2
    assert collection.events["source"].nunique() == 2
    assert collection.events["source"].str.startswith("gdelt:flashpoint-").all()


def test_scorer_identity_is_bound_into_source_policy(tmp_path: Path) -> None:
    first = collect_live_gdelt_global_events(
        _request(),
        tmp_path / "scorer-a",
        scorer=_Scorer(),
        scorer_identity="finbert@revision-a",
        fetch=lambda _: _fetch_result([]),
        clock=_Clock([END, FETCHED, COMPLETED]),
    )
    second = collect_live_gdelt_global_events(
        _request(),
        tmp_path / "scorer-b",
        scorer=_Scorer(),
        scorer_identity="finbert@revision-b",
        fetch=lambda _: _fetch_result([]),
        clock=_Clock([END, FETCHED, COMPLETED]),
    )

    first_request = first.manifest["request"]
    second_request = second.manifest["request"]
    assert isinstance(first_request, Mapping)
    assert isinstance(second_request, Mapping)
    first_policy = first_request["source_policy_sha256"]
    second_policy = second_request["source_policy_sha256"]
    assert first_policy != second_policy


def _request(
    *,
    queries: tuple[str, ...] = QUERIES,
    max_records: int = 100,
) -> GdeltDocumentRequest:
    return GdeltDocumentRequest(
        queries=queries,
        requested_start_utc=START,
        requested_end_utc=END,
        max_records=max_records,
    )


def _http_response(
    status_code: int,
    payload: object | None = None,
    *,
    raw: bytes | None = None,
    headers: Mapping[str, str] | None = None,
) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.url = GDELT_DOCUMENT_ENDPOINT
    response.headers.update(headers or {})
    response.headers.setdefault("Content-Type", "application/json")
    response._content = raw if raw is not None else json.dumps(payload).encode("utf-8")
    return response


def _article(
    *,
    title: str,
    url: str,
    published_at: str,
    **extra: object,
) -> Mapping[str, object]:
    return {
        "title": title,
        "url": url,
        "seendate": published_at,
        "domain": "example.com",
        "summary": "Global macro event",
        **extra,
    }
