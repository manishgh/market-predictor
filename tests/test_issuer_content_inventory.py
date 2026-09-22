"""Original query chunks written by the real Alpaca producer; no provider or archive access."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest

from market_predictor.canonical.audits import CanonicalAuditReport, audit_canonical_events
from market_predictor.canonical.normalize import canonicalize_events
from market_predictor.canonical.store import manifest_path_for, write_canonical_artifact
from market_predictor.catalysts.issuer_events import alpaca_news_collection as producer
from market_predictor.catalysts.issuer_events import content_inventory as inventory
from market_predictor.catalysts.issuer_events.news_history_contracts import NEWS_HISTORY_REQUEST_SCHEMA
from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.data_quality import _safe_json
from market_predictor.evidence.hashing import json_sha256
from market_predictor.sources.alpaca import AlpacaNewsPage

SECURITY = "query:issuer"
UNIT_START, UNIT_END = datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
COLLECTION = Path("data/raw/news")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _raw_hash(item: dict[str, Any]) -> str:
    return _sha(_safe_json(item).encode("utf-8"))


def _article(**changes: Any) -> dict[str, Any]:
    return {"id": 1, "headline": "Issuer report", "content": "Body", "summary": "Summary",
            "created_at": "2020-03-01T12:00:00Z", "updated_at": "2020-03-01T12:00:01Z",
            "source": "provider", "url": "https://example.test", "symbols": ["BRK.B"], **changes}


def _build(root: Path, monkeypatch: pytest.MonkeyPatch, pages: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Run the producer's own work-unit, page, normalization and canonical writers."""
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(root)
    memberships = pd.DataFrame({"security_id": [SECURITY], "ticker": ["BRK.B"],
        "effective_from_utc": [pd.Timestamp(UNIT_START)], "effective_to_utc": [pd.Timestamp(UNIT_END)]})
    [unit] = producer._build_work_units(memberships, start_utc=UNIT_START, end_exclusive_utc=UNIT_END,
                                        chunk_days=366, provider_symbol_for=lambda ticker: ticker.replace("-", "."))
    request = {"schema": NEWS_HISTORY_REQUEST_SCHEMA, "start_utc": UNIT_START.isoformat(),
               "end_exclusive_utc": UNIT_END.isoformat(), "availability_policy": "provider_publication_proxy",
               "include_content": True, "page_limit": 50, "work_units": [producer._work_unit_record(unit)]}
    request_hash = producer._sha256_json(request)
    producer._write_or_validate_request(COLLECTION / "_request.json", request, request_hash)

    def fetch(_symbol: str, _start: datetime, _end: datetime, token: str | None) -> AlpacaNewsPage:
        index = 0 if token is None else int(token)
        return AlpacaNewsPage(request_page_token=token, news=tuple(pages[index]),
                              next_page_token=str(index + 1) if index < len(pages) - 1 else None)

    payloads = producer._collect_pages(unit=unit, page_dir=COLLECTION / "raw_pages" / unit.chunk_id,
        request_hash=request_hash, fetch_page=fetch, memory_budget_gib=5.0, memory_headroom_gib=0.75,
        memory_check=lambda: None)
    raw, page_inputs, _ = producer._normalize_pages(unit, payloads)
    events = canonicalize_events(raw, availability_policy="provider_publication_proxy")
    artifact = COLLECTION / "events" / f"{unit.chunk_id}.parquet"
    write_canonical_artifact(events, artifact, artifact_type="events",
        audit=CanonicalAuditReport(checks=audit_canonical_events(events, require_observed=False)),
        inputs={"collection_request_sha256": request_hash, "chunk_id": unit.chunk_id, **page_inputs},
        production_ready=False)
    args = {"root": root, "security_id": SECURITY, "ticker": "BRK.B",
            "start_utc": pd.Timestamp("2020-01-01T00:00:00Z"), "cutoff_utc": pd.Timestamp("2020-12-31T23:59:59Z"),
            "memory_check": lambda: None}
    _pin(args, artifact)
    return args


def _pin(args: dict[str, Any], artifact: Path) -> None:
    sidecar = manifest_path_for(artifact)
    args["event_artifact"] = SourcePin(path=artifact.as_posix(), sha256=_sha((args["root"] / artifact).read_bytes()))
    args["event_manifest"] = SourcePin(path=sidecar.as_posix(), sha256=_sha((args["root"] / sidecar).read_bytes()))


def _artifact(args: dict[str, Any]) -> Path:
    return Path(args["root"], args["event_artifact"].path)


def _sidecar(args: dict[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = json.loads((Path(args["root"]) / args["event_manifest"].path).read_bytes())
    return value


def _write_sidecar(args: dict[str, Any], manifest: dict[str, Any]) -> None:
    (Path(args["root"]) / args["event_manifest"].path).write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    _pin(args, Path(args["event_artifact"].path))


def _repin_rows(args: dict[str, Any], change: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    artifact = _artifact(args)
    frame = change(pd.read_parquet(artifact))
    frame.to_parquet(artifact, index=False)
    manifest = _sidecar(args)
    manifest.update(artifact_sha256=_sha(artifact.read_bytes()), rows=len(frame))
    _write_sidecar(args, manifest)


def _page_path(args: dict[str, Any], index: int = 0) -> Path:
    chunk = _sidecar(args)["inputs"]["chunk_id"]
    return Path(args["root"], COLLECTION, "raw_pages", str(chunk), f"page_{index:06d}.json")


def _repin_page(args: dict[str, Any], change: Callable[[dict[str, Any]], None], *, index: int = 0,
                envelope: bool = True) -> None:
    path = _page_path(args, index)
    page = json.loads(path.read_bytes())
    change(page)
    if envelope:
        page["content_sha256"] = json_sha256({k: v for k, v in page.items() if k != "content_sha256"})
    path.write_text(json.dumps(page, sort_keys=True), encoding="utf-8")
    manifest = _sidecar(args)
    key = next(key for key in manifest["inputs"] if key.replace("\\", "/").endswith(path.name))
    manifest["inputs"][key] = _sha(path.read_bytes())
    _write_sidecar(args, manifest)


def _rebind_request(args: dict[str, Any], change: Callable[[dict[str, Any]], None]) -> None:
    """Consistently rehash the request and every child that declares it."""
    path = Path(args["root"]) / COLLECTION / "_request.json"
    body = json.loads(path.read_bytes())
    body.pop("request_sha256")
    change(body)
    digest = json_sha256(body)
    path.write_text(json.dumps({**body, "request_sha256": digest}, sort_keys=True), encoding="utf-8")
    manifest = _sidecar(args)
    manifest["inputs"]["collection_request_sha256"] = digest
    _write_sidecar(args, manifest)
    index = 0
    while _page_path(args, index).exists():
        _repin_page(args, lambda page: page.update(collection_request_sha256=digest), index=index)
        index += 1


def _inspect(args: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    return inventory.inspect_saved_alpaca_content(**args)


@pytest.mark.parametrize("fields,category,state", [({}, "provider_body_field", "nonempty"),
    ({"content": ""}, "provider_summary_field", "empty"),
    ({"content": "  "}, "provider_summary_field", "blank"),
    ({"content": 5}, "provider_summary_field", "non_text"),
    ({"content": None, "summary": ""}, "headline_only", "null"),
    ({"content": "Issuer report"}, "provider_body_field", "nonempty")])
def test_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fields: dict[str, Any], category: str,
                state: str) -> None:
    frame, summary = _inspect(_build(tmp_path, monkeypatch, [[_article(**fields)]]))
    row = frame.iloc[0]
    assert row.content_category == category and row.body_state == state
    assert row.source_version_sha256 == row.raw_sha256 and row.provider_story_id == "1"
    assert row.query_security_id == SECURITY and row.query_ticker == "BRK-B"
    assert row.inventory_status == "included" and row.availability_policy == "provider_publication_proxy"
    assert row.available_at_utc == pd.Timestamp("2020-03-01T12:00:01Z")
    assert row.attribution_status == "not_established_by_content_inventory"
    assert bool(row.body_equals_title) == (fields.get("content") == "Issuer report")
    for flag in ("production_eligible", "training_eligible", "serving_eligible"):
        assert not row[flag] and summary[flag] is False
    assert summary["include_content_requested"] is True and summary["query_provider_symbol"] == "BRK.B"
    assert summary["category_counts"]["included"][category] == 1


def test_absent_unicode_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    item = _article(headline="Café report")
    del item["content"], item["summary"]
    frame, _ = _inspect(_build(tmp_path, monkeypatch, [[item]]))
    row = frame.iloc[0]
    assert row.body_state == row.summary_state == "absent"
    assert row.raw_sha256 == _raw_hash(item) != json_sha256(item)
    assert row.chosen_field_sha256 == _sha(item["headline"].encode())


def test_revisions_duplicates_and_all_occurrences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old, current = _article(content="Old", updated_at="2020-03-01T12:00:00Z"), _article()
    args = _build(tmp_path, monkeypatch, [[old, current], [current]])
    frame, summary = _inspect(args)
    assert frame.iloc[0].raw_sha256 == _raw_hash(current)
    locations = json.loads(frame.iloc[0].raw_record_locators_json)
    assert [row["news_index"] for row in locations] == [1, 0]
    assert [row["page_path"].rsplit("/", 1)[1] for row in locations] == ["page_000000.json", "page_000001.json"]
    assert summary["query_chunk_provider_records"] == 3 and summary["query_chunk_admitted_stories"] == 1
    assert summary["query_chunk_admitted_versions"] == 2
    assert summary["query_chunk_superseded_versions"] == summary["query_chunk_duplicate_version_occurrences"] == 1


def test_canonical_older_revision_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old, current = _article(content="Old", updated_at="2020-03-01T12:00:00Z"), _article()
    args = _build(tmp_path, monkeypatch, [[old, current]])

    def keep_old(frame: pd.DataFrame) -> pd.DataFrame:
        stamp = pd.Timestamp(old["updated_at"])
        return frame.assign(raw_sha256=_raw_hash(old), text="Old", provider_updated_at_utc=stamp,
                            available_at_utc=stamp, feature_available_at_utc=stamp)

    _repin_rows(args, keep_old)
    with pytest.raises(DataReadinessError, match="producer-retained"):
        _inspect(args)


def test_first_seen_must_come_from_retained_occurrence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _build(tmp_path, monkeypatch, [[_article()], [_article()]])
    earlier = "2025-01-01T00:00:00+00:00"
    _repin_page(args, lambda page: page.update(collected_at_utc=earlier), index=0)
    assert _inspect(args)[1]["included_rows"] == 1
    _repin_rows(args, lambda frame: frame.assign(first_seen_at_utc=pd.Timestamp(earlier)))
    with pytest.raises(DataReadinessError, match="producer-retained"):
        _inspect(args)


def test_producer_discarded_raw_items_are_counted_not_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    discarded = [_article(id=None, created_at="not-a-date"), _article(id=3, created_at="2019-06-01T00:00:00Z"),
                 _article(id=4, symbols=["MSFT"]), _article(id=5, headline=" "),
                 _article(id=6, created_at="3000-01-01T00:00:00Z", updated_at=None)]
    frame, summary = _inspect(_build(tmp_path, monkeypatch, [[_article(), *discarded]]))
    assert len(frame) == 1 and summary["query_chunk_provider_records"] == 6
    assert summary["query_chunk_discarded_records"] == {"clock": 1, "window": 2, "symbol": 1, "title": 1}


@pytest.mark.parametrize("change", [{"headline": ""}, {"symbols": ["MSFT"]}, {"created_at": "2019-06-01T00:00:00Z"}])
def test_canonical_row_for_producer_discarded_item_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any],
) -> None:
    args = _build(tmp_path, monkeypatch, [[_article()]])
    item = _article(**change)
    _repin_page(args, lambda page: page.update(news=[item]))
    _repin_rows(args, lambda frame: frame.assign(raw_sha256=_raw_hash(item)))
    with pytest.raises(DataReadinessError, match="discarded"):
        _inspect(args)


def test_inventory_status_and_window_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items = [_article(id=1), _article(id=2, created_at="2020-06-01T00:00:00Z", updated_at="2020-07-15T00:00:00Z"),
             _article(id=3, created_at="2020-08-01T00:00:00Z", updated_at=None, content="")]
    args = _build(tmp_path, monkeypatch, [items])
    args["cutoff_utc"] = pd.Timestamp("2020-06-30T00:00:00Z")
    frame, summary = _inspect(args)
    assert dict(zip(frame.provider_story_id, frame.inventory_status, strict=True)) == {
        "1": "included", "2": "version_after_cutoff"}
    assert (summary["included_rows"], summary["version_after_cutoff_rows"], summary["outside_publication_window_rows"]) == (1, 1, 1)
    assert summary["recorded_rows"] == 2 and summary["canonical_rows"] == 3
    assert summary["category_counts"]["version_after_cutoff"]["provider_body_field"] == 1


def test_cutoff_is_inclusive_to_the_nanosecond(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _build(tmp_path, monkeypatch, [[_article()]])
    args["cutoff_utc"] = pd.Timestamp("2020-03-01T12:00:01Z")
    assert _inspect(args)[1]["included_rows"] == 1
    args["cutoff_utc"] -= pd.Timedelta(1, "ns")
    assert _inspect(args)[1]["version_after_cutoff_rows"] == 1
    args["cutoff_utc"] = pd.Timestamp("2020-03-01T11:59:59.999999999Z")
    assert _inspect(args)[1]["outside_publication_window_rows"] == 1
    args["start_utc"] = pd.Timestamp("2020-01-01")
    with pytest.raises(DataReadinessError):
        _inspect(args)


def test_empty_selection_and_parquet_types(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _build(tmp_path / "empty", monkeypatch, [[_article()]])
    args["start_utc"] = pd.Timestamp("2020-06-01T00:00:00Z")
    empty, summary = _inspect(args)
    full, _ = _inspect(_build(tmp_path / "full", monkeypatch, [[_article()]]))
    assert empty.empty and empty.dtypes.equals(full.dtypes)
    assert summary["outside_publication_window_rows"] == 1 and not summary["known_empty_coverage"]
    assert sum(sum(counts.values()) for counts in summary["category_counts"].values()) == 0
    empty.to_parquet(tmp_path / "empty.parquet", index=False)
    assert pd.read_parquet(tmp_path / "empty.parquet").dtypes.equals(empty.dtypes)


def test_batch_size_does_not_change_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _build(tmp_path, monkeypatch, [[_article(id=index, headline=f"Report {index}") for index in range(1, 8)]])
    frame, summary = _inspect(args)
    monkeypatch.setattr(inventory, "EVENT_BATCH_ROWS", 3)
    other_frame, other_summary = _inspect(args)
    pd.testing.assert_frame_equal(frame, other_frame)
    assert summary == other_summary and len(frame) == 7
    assert summary["query_chunk_admitted_stories_without_canonical_row"] == 0


def test_output_is_checkout_independent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _build(tmp_path / "first", monkeypatch, [[_article()]])
    shutil.copytree(tmp_path / "first", tmp_path / "second")
    frame, summary = _inspect(args)
    other_frame, other_summary = _inspect({**args, "root": tmp_path / "second"})
    pd.testing.assert_frame_equal(frame, other_frame)
    assert summary == other_summary
    assert all(not Path(path).is_absolute() and "\\" not in path for path in summary["source_files"])
    assert "_request.json" in "".join(summary["source_files"])


def test_memory_check_runs_and_stops_inspection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _build(tmp_path, monkeypatch, [[_article(id=1), _article(id=2)]])
    calls: list[None] = []
    args["memory_check"] = lambda: calls.append(None)
    _inspect(args)
    assert len(calls) > 2 * 2 * 2

    def exhausted() -> None:
        raise MemoryBudgetError("test-only memory pressure")

    args["memory_check"] = exhausted
    with pytest.raises(MemoryBudgetError):
        _inspect(args)


@pytest.mark.parametrize("limit", ["MAX_FILE_BYTES", "MAX_EVENT_ROWS", "MAX_RAW_PAGE_BYTES", "MAX_PARQUET_UNCOMPRESSED_BYTES"])
def test_preflight_memory_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str) -> None:
    args = _build(tmp_path, monkeypatch, [[_article()]])
    monkeypatch.setattr(inventory, limit, 0)
    with pytest.raises(DataReadinessError):
        _inspect(args)


def test_combined_pages_not_only_individual_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _build(tmp_path, monkeypatch, [[_article()], [_article()]])
    first, second = _page_path(args, 0), _page_path(args, 1)
    monkeypatch.setattr(inventory, "MAX_RAW_PAGE_BYTES", max(first.stat().st_size, second.stat().st_size))
    with pytest.raises(DataReadinessError, match="combined raw pages"):
        _inspect(args)


@pytest.mark.parametrize("change,message", [
    ({"schema": "alpaca.news_http_receipt.v1"}, "schema/request/chunk"), ({"chunk_id": "wrong"}, "schema/request/chunk"),
    ({"collection_request_sha256": "b" * 64}, "schema/request/chunk"), ({"request_page_token": "wrong"}, "pagination request"),
    ({"next_page_token": "unterminated"}, "early-terminal"), ({"next_page_token": ""}, "invalid page token"),
    ({"page_index": True}, "page sequence"), ({"news": {}}, "array")])
def test_rehashed_envelope_poison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any],
                                  message: str) -> None:
    args = _build(tmp_path, monkeypatch, [[_article()]])
    _repin_page(args, lambda page: page.update(change))
    with pytest.raises(DataReadinessError, match=message):
        _inspect(args)


@pytest.mark.parametrize("change,message", [
    (lambda page: page.pop("next_page_token"), "terminal evidence"),
    (lambda page: page.update(next_page_token="1"), "token cycle")])
def test_pagination_evidence_poison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                    change: Callable[[dict[str, Any]], None], message: str) -> None:
    args = _build(tmp_path, monkeypatch, [[_article()], [_article()]])
    _repin_page(args, change, index=1)
    with pytest.raises(DataReadinessError, match=message):
        _inspect(args)


def test_page_envelope_hash_is_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _build(tmp_path, monkeypatch, [[_article()]])
    _repin_page(args, lambda page: page.update(collected_at_utc="2025-01-01T00:00:00+00:00"), envelope=False)
    with pytest.raises(DataReadinessError, match="envelope hash"):
        _inspect(args)


@pytest.mark.parametrize("which", ["event_artifact", "event_manifest", "page"])
def test_physical_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str) -> None:
    args = _build(tmp_path, monkeypatch, [[_article()]])
    path = _page_path(args) if which == "page" else tmp_path / args[which].path
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(DataReadinessError, match="hash mismatch"):
        _inspect(args)


@pytest.mark.parametrize("change,message", [
    (lambda body: body.update(include_content=False), "semantics unsupported"),
    (lambda body: body.update(availability_policy="observed"), "semantics unsupported"),
    (lambda body: body.update(schema="swing.alpaca_news_history_request.v0"), "semantics unsupported"),
    (lambda body: body["work_units"][0].update(security_id="query:other"), "does not derive"),
    (lambda body: body["work_units"].append(dict(body["work_units"][0])), "exactly one"),
    (lambda body: body["work_units"][0].pop("provider_symbol"), "malformed request work unit")])
def test_request_work_unit_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                   change: Callable[[dict[str, Any]], None], message: str) -> None:
    args = _build(tmp_path, monkeypatch, [[_article()]])
    _rebind_request(args, change)
    with pytest.raises(DataReadinessError, match=message):
        _inspect(args)


def test_request_content_hash_is_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _build(tmp_path, monkeypatch, [[_article()]])
    path = tmp_path / COLLECTION / "_request.json"
    path.write_bytes(path.read_bytes() + b"\n")
    assert _inspect(args)[1]["included_rows"] == 1
    payload = json.loads(path.read_bytes())
    payload["page_limit"] = 49
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataReadinessError, match="request hash mismatch"):
        _inspect(args)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        _inspect(args)


@pytest.mark.parametrize("field,value", [("security_id", "query:other"), ("ticker", "MSFT")])
def test_query_arguments_must_match_request_work_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      field: str, value: str) -> None:
    args = _build(tmp_path, monkeypatch, [[_article()]])
    args[field] = value
    with pytest.raises(DataReadinessError, match="differs from its request work unit"):
        _inspect(args)


@pytest.mark.parametrize("column,value,message", [
    ("raw_sha256", "0" * 64, "no original provider object"), ("security_id", "wrong", "query identity"),
    ("ticker", "MSFT", "query identity"), ("event_id", "0" * 64, "event identity"),
    ("title", "wrong", "fallback"), ("text", "wrong", "fallback"), ("summary", "wrong", "fallback"),
    ("url", "wrong", "fallback"), ("source", "alpaca:wrong", "fallback"),
    ("availability_policy", "observed", "publication proxy"),
    ("available_at_utc", pd.Timestamp("2020-03-02T12:00:01Z"), "publication proxy"),
    ("available_at_utc", "first_seen", "publication proxy"),
    ("first_seen_at_utc", pd.Timestamp("2025-01-01T00:00:00Z"), "producer-retained")])
def test_canonical_poison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, column: str, value: Any, message: str) -> None:
    args = _build(tmp_path, monkeypatch, [[_article()]])
    _repin_rows(args, lambda frame: frame.assign(**{column: frame.first_seen_at_utc if value == "first_seen" else value}))
    with pytest.raises(DataReadinessError, match=message):
        _inspect(args)


def test_provider_publication_clock_is_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _build(tmp_path, monkeypatch, [[_article(updated_at=None)]])
    shift = pd.Timedelta(1, "s")
    _repin_rows(args, lambda frame: frame.assign(published_at_utc=frame.published_at_utc + shift,
                                                 available_at_utc=frame.available_at_utc + shift))
    with pytest.raises(DataReadinessError, match="provider clocks mismatch"):
        _inspect(args)


def test_duplicate_canonical_event_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _build(tmp_path, monkeypatch, [[_article()]])
    _repin_rows(args, lambda frame: pd.concat([frame, frame], ignore_index=True))
    with pytest.raises(DataReadinessError, match="duplicate canonical event IDs"):
        _inspect(args)


@pytest.mark.parametrize("case,message", [
    ("escape", "escapes"), ("page_name", "raw-page path"), ("duplicate_page", "duplicate declared pages"),
    ("artifact_path", "declared artifact path"), ("artifact_hash", "artifact declaration"),
    ("chunk", "chunk identity"), ("request_identity", "collection request identity")])
def test_sidecar_declaration_poison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, message: str) -> None:
    args = _build(tmp_path, monkeypatch, [[_article()]])
    manifest = _sidecar(args)
    inputs = manifest["inputs"]
    page_key = next(key for key in inputs if key.endswith(".json"))
    if case == "escape":
        inputs["../outside/page_000000.json"] = inputs.pop(page_key)
    elif case == "page_name":
        inputs[page_key.replace("page_000000", "page_0")] = inputs.pop(page_key)
    elif case == "duplicate_page":
        inputs[page_key.replace("page_000000", "./page_000000")] = inputs[page_key]
    elif case == "artifact_path":
        manifest["artifact_path"] = "data/raw/news/events/other.parquet"
    elif case == "artifact_hash":
        manifest["artifact_sha256"] = "0" * 64
    elif case == "chunk":
        inputs["chunk_id"] = "chunk"
    else:
        inputs["collection_request_sha256"] = "not-a-hash"
    _write_sidecar(args, manifest)
    with pytest.raises(DataReadinessError, match=message):
        _inspect(args)


@pytest.mark.parametrize("sidecar", ["data/raw/news/events/other.parquet.manifest.json", "data/raw/news/other.manifest.json"])
def test_sidecar_must_belong_to_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sidecar: str) -> None:
    args = _build(tmp_path, monkeypatch, [[_article()]])
    target = tmp_path / sidecar
    shutil.copyfile(tmp_path / args["event_manifest"].path, target)
    args["event_manifest"] = SourcePin(path=sidecar, sha256=_sha(target.read_bytes()))
    with pytest.raises(DataReadinessError, match="sidecar path"):
        _inspect(args)


@pytest.mark.parametrize("excluded", ["outside", "future"])
@pytest.mark.parametrize("poison", ["hash", "text", "first_seen"])
def test_excluded_rows_still_verify_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, excluded: str,
                                               poison: str) -> None:
    args = _build(tmp_path, monkeypatch, [[_article()]])
    args["cutoff_utc"] = pd.Timestamp("2020-02-01T00:00:00Z" if excluded == "outside" else "2020-03-01T12:00:00Z")
    assert _inspect(args)[1]["recorded_rows" if excluded == "future" else "outside_publication_window_rows"] == 1
    values = {"hash": ("raw_sha256", "0" * 64), "text": ("text", "Not the saved provider text"),
              "first_seen": ("first_seen_at_utc", pd.Timestamp("2025-01-01T00:00:00Z"))}
    column, value = values[poison]
    _repin_rows(args, lambda frame: frame.assign(**{column: value}))
    with pytest.raises(DataReadinessError):
        _inspect(args)


def test_dictionary_expansion_cumulative_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items = [_article(id=index, headline=f"Report {index}", content="x" * 4096) for index in range(1, 41)]
    args = _build(tmp_path, monkeypatch, [items])
    parquet = pq.ParquetFile(_artifact(args))  # type: ignore[no-untyped-call]
    metadata_bytes = sum(parquet.metadata.row_group(group).column(column).total_uncompressed_size
                         for group in range(parquet.metadata.num_row_groups)
                         for column in range(parquet.metadata.num_columns))
    batches = list(parquet.iter_batches(  # type: ignore[no-untyped-call]
        batch_size=1, columns=list(inventory._EVENT_COLUMNS), use_threads=False))
    limit = metadata_bytes + max(batch.nbytes for batch in batches)
    assert sum(batch.nbytes for batch in batches) > limit
    assert max(batch.nbytes for batch in batches) < limit
    calls: list[None] = []
    args["memory_check"] = lambda: calls.append(None)
    monkeypatch.setattr(inventory, "MAX_DECODED_EVENT_BYTES", limit)
    with pytest.raises(DataReadinessError, match="decoded Parquet cumulative"):
        _inspect(args)
    assert len(calls) > len(items) * 2


def test_decoded_size_checked_before_row_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    class OversizedBatch:
        nbytes = 11

        def to_pylist(self) -> list[dict[str, Any]]:
            raise AssertionError("oversized batch reached row conversion")

    class ParquetProbe:
        def iter_batches(self, **kwargs: Any) -> Any:
            assert kwargs["batch_size"] == 1
            return iter([OversizedBatch()])

    monkeypatch.setattr(inventory, "MAX_DECODED_BATCH_BYTES", 10)
    probe: Any = ParquetProbe()
    with pytest.raises(DataReadinessError, match="decoded Parquet batch"):
        list(inventory._event_batches(probe, ["text"], lambda: None))
