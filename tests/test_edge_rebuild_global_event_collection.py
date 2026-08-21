from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
import requests

from market_predictor.canonical.store import load_canonical_artifact
from market_predictor.edge_rebuild import global_event_collection as collection_module
from market_predictor.edge_rebuild.global_event_authority import publish_global_event_authority
from market_predictor.edge_rebuild.global_event_collection import (
    GdeltCollectionRequest,
    GdeltFetchResult,
    collect_live_gdelt_global_events,
    fetch_gdelt_doc_api,
    load_gdelt_global_event_collection,
    validate_gdelt_collection_request,
)
from market_predictor.core.errors import DataReadinessError

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
) -> GdeltFetchResult:
    tagged_records = [
        {**dict(record), "collection_query": record.get("collection_query", QUERIES[0])}
        for record in records
    ]
    return GdeltFetchResult(
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
            GdeltFetchResult(
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
    fetch_result: GdeltFetchResult,
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


def test_doc_api_uses_exact_params_and_preserves_query_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Mapping[str, object], float]] = []
    responses = [
        _http_response(
            200,
            {
                "articles": [
                    _article(
                        title=f"Article {index}",
                        url=f"https://example.com/{index}",
                        published_at="2025-01-10T19:30:00Z",
                    )
                ]
            },
        )
        for index in range(len(QUERIES))
    ]

    def fake_get(
        url: str,
        *,
        params: Mapping[str, object],
        timeout: float,
    ) -> requests.Response:
        calls.append((url, params, timeout))
        return responses.pop(0)

    monkeypatch.setattr(requests, "get", fake_get)
    result = fetch_gdelt_doc_api(_request())

    assert result.complete is True
    assert result.completed_queries == QUERIES
    assert [record["collection_query"] for record in result.records] == list(QUERIES)
    assert len(calls) == len(QUERIES)
    for index, (url, params, timeout) in enumerate(calls):
        assert url == collection_module.GDELT_DOC_ENDPOINT
        assert params == {
            "query": QUERIES[index],
            "mode": "artlist",
            "format": "json",
            "maxrecords": 100,
            "startdatetime": "20250107190000",
            "enddatetime": "20250110200000",
        }
        assert timeout == 30.0


@pytest.mark.parametrize("transport_error", [requests.Timeout(), requests.ConnectionError()])
def test_doc_api_retries_transport_errors_and_retryable_statuses(
    monkeypatch: pytest.MonkeyPatch,
    transport_error: requests.RequestException,
) -> None:
    outcomes: list[requests.Response | requests.RequestException] = [
        transport_error,
        _http_response(429, {}, headers={"Retry-After": "1.5"}),
        _http_response(503, {}),
        _http_response(200, {"articles": []}),
    ]
    sleeps: list[float] = []

    def fake_get(
        _url: str,
        *,
        params: Mapping[str, object],
        timeout: float,
    ) -> requests.Response:
        assert params["query"] == QUERIES[0]
        assert timeout == 30.0
        outcome = outcomes.pop(0)
        if isinstance(outcome, requests.RequestException):
            raise outcome
        return outcome

    monkeypatch.setattr(requests, "get", fake_get)
    result = fetch_gdelt_doc_api(
        _request(queries=(QUERIES[0],)),
        max_attempts=4,
        initial_backoff_seconds=0.25,
        sleep=sleeps.append,
    )

    assert result.records == []
    assert sleeps == [0.25, 1.5, 1.0]
    assert not outcomes


def test_doc_api_rejects_invalid_json_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_get(
        _url: str,
        *,
        params: Mapping[str, object],
        timeout: float,
    ) -> requests.Response:
        nonlocal calls
        calls += 1
        return _http_response(200, raw=b"not-json")

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(DataReadinessError, match="invalid JSON"):
        fetch_gdelt_doc_api(_request(queries=(QUERIES[0],)))
    assert calls == 1


def test_doc_api_marks_maximum_size_response_as_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    articles = [
        _article(
            title=f"Article {index}",
            url=f"https://example.com/{index}",
            published_at="2025-01-10T19:30:00Z",
        )
        for index in range(2)
    ]
    monkeypatch.setattr(
        requests,
        "get",
        lambda *_args, **_kwargs: _http_response(200, {"articles": articles}),
    )

    result = fetch_gdelt_doc_api(
        _request(queries=(QUERIES[0],), max_records=2),
    )
    assert result.complete is False
    assert len(result.records) == 2


def test_doc_api_rejects_permanent_http_failure_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_get(
        _url: str,
        *,
        params: Mapping[str, object],
        timeout: float,
    ) -> requests.Response:
        nonlocal calls
        calls += 1
        return _http_response(400, {"error": "bad request"})

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(DataReadinessError, match="permanently.*400"):
        fetch_gdelt_doc_api(_request(queries=(QUERIES[0],)))
    assert calls == 1


def test_doc_api_stops_after_bounded_retryable_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_get(
        _url: str,
        *,
        params: Mapping[str, object],
        timeout: float,
    ) -> requests.Response:
        nonlocal calls
        calls += 1
        return _http_response(503, {"error": "temporarily unavailable"})

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(DataReadinessError, match="after 3 attempts.*503"):
        fetch_gdelt_doc_api(
            _request(queries=(QUERIES[0],)),
            max_attempts=3,
            initial_backoff_seconds=0.25,
            sleep=sleeps.append,
        )
    assert calls == 3
    assert sleeps == [0.25, 0.5]


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


def test_public_request_validation_runs_without_a_scorer() -> None:
    normalized = validate_gdelt_collection_request(_request())
    assert normalized.queries == QUERIES
    with pytest.raises(ValueError, match="reversed"):
        validate_gdelt_collection_request(
            GdeltCollectionRequest(
                queries=QUERIES,
                requested_start_utc=END,
                requested_end_utc=START,
            )
        )


def _request(
    *,
    queries: tuple[str, ...] = QUERIES,
    max_records: int = 100,
) -> GdeltCollectionRequest:
    return GdeltCollectionRequest(
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
    response.url = collection_module.GDELT_DOC_ENDPOINT
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
