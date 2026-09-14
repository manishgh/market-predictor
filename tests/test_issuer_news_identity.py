"""Synthetic identity proofs; no provider reads, publication or source rescoring."""
from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from typing import Any

import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.universe import issuer_news_identity as identity
from market_predictor.universe.issuer_news_identity import (
    BRIDGE_COLUMNS,
    build_issuer_news_bridge,
    map_news_coverage,
    map_news_relations,
)

OLD = "cik:0000320193:ticker:AAPL"
NEW = "cik:0000320193"
PIN = "a" * 64


def _time(day: int) -> pd.Timestamp:
    return pd.Timestamp(f"2020-01-{day:02d}T00:00:00Z")


def _row(security: str = OLD, ticker: str = "AAPL", *, start: int = 1,
         end: int | None = None, available: int = 1, sec: str | None = None) -> dict[str, Any]:
    row = {"security_id": security, "ticker": ticker, "effective_from_utc": _time(start),
        "effective_to_utc": pd.NaT if end is None else _time(end), "available_at_utc": _time(available)}
    if sec is not None:
        row["sec_cik"] = sec
    return row


def _inputs() -> list[pd.DataFrame]:
    return [pd.DataFrame([_row()]), pd.DataFrame([_row()]), pd.DataFrame([_row(NEW)]),
        pd.DataFrame([_row(NEW, sec="0000320193")])]


def _bridge() -> pd.DataFrame:
    return build_issuer_news_bridge(*_inputs(), evidence_sha256=PIN)


def _relations() -> pd.DataFrame:
    return pd.DataFrame([{
        "relation_id": f"relation-{day}", "event_id": f"event-{day}", "target_security_id": OLD,
        "target_ticker": "AAPL", "source_security_id": OLD, "source_ticker": "AAPL",
        "event_feature_available_at_utc": _time(day), "identity_available_at_utc": _time(1),
        "feature_available_at_utc": _time(20), "sentiment_numeric": 0.23456789,
        "relation_channel": "direct_issuer", "relation_score": 1.0,
        "preparation_original_chunk_id": "physical-child", "preparation_source_child_set_sha256": PIN,
    } for day in (1, 5, 10, 15)])


def _ledger() -> pd.DataFrame:
    return pd.DataFrame([{"chunk_id": "old-chunk", "security_id": OLD, "ticker": "AAPL",
        "requested_start_utc": _time(1), "requested_end_utc": _time(20),
        "status": "unavailable_saved_evidence", "coverage_blindspot": True, "row_count": 0,
        "preparation_original_chunk_id": "physical-child", "preparation_source_child_set_sha256": PIN}])


def test_aapl_requires_four_intervals_and_maximum_availability() -> None:
    inputs = _inputs()
    for frame, start, end, available in zip(inputs, (1, 3, 4, 5), (19, 18, 17, 16), (2, 3, 6, 4), strict=True):
        frame.loc[0, "effective_from_utc"] = _time(start)
        frame["effective_to_utc"] = pd.Series([_time(end)])
        frame.loc[0, "available_at_utc"] = _time(available)
    original = [frame.copy(deep=True) for frame in inputs]
    bridge = build_issuer_news_bridge(*inputs, evidence_sha256=PIN)
    assert tuple(bridge.columns) == BRIDGE_COLUMNS
    assert bridge.iloc[0].source_security_id == OLD
    assert bridge.iloc[0].target_security_id == NEW
    assert bridge.iloc[0].effective_from_utc == _time(5)
    assert bridge.iloc[0].effective_to_utc == _time(16)
    assert bridge.iloc[0].available_at_utc == _time(6)
    for before, after in zip(original, inputs, strict=True):
        pd.testing.assert_frame_equal(before, after)


def test_open_interval_and_pin_bound_hash_are_deterministic() -> None:
    first = _bridge()
    assert pd.isna(first.iloc[0].effective_to_utc)
    pd.testing.assert_frame_equal(first, _bridge())
    second = build_issuer_news_bridge(*_inputs(), evidence_sha256="b" * 64)
    assert first.iloc[0].bridge_row_sha256 != second.iloc[0].bridge_row_sha256
    assert len(first.iloc[0].bridge_row_sha256) == 64


def test_open_ended_bridge_preserves_exact_parquet_resume_representation() -> None:
    bridge = _bridge()
    with BytesIO() as payload:
        bridge.to_parquet(payload, index=False)
        payload.seek(0)
        restored = pd.read_parquet(payload)
    pd.testing.assert_frame_equal(restored, bridge, check_exact=True)
    pd.testing.assert_frame_equal(map_news_relations(_relations(), restored), map_news_relations(_relations(), bridge))


def test_nullable_stored_identity_clock_accepts_proven_utc_time_without_backdating() -> None:
    relations = _relations()
    relations["identity_available_at_utc"] = pd.Series(pd.NaT, index=relations.index, dtype="datetime64[ms]")
    result = map_news_relations(relations, _bridge())
    assert result.identity_original_identity_available_at_utc.isna().all()
    assert result.identity_available_at_utc.eq(_time(1)).all()
    assert result.feature_available_at_utc.eq(_time(20)).all()
    assert result.target_security_id.eq(NEW).all()


def test_string_clock_mapping_retains_strict_aware_validation() -> None:
    relations = _relations()
    relations["feature_available_at_utc"] = relations.feature_available_at_utc.map(lambda value: value.isoformat())
    relations["identity_available_at_utc"] = relations.identity_available_at_utc.map(lambda value: value.isoformat())
    assert map_news_relations(relations, _bridge()).target_security_id.eq(NEW).all()
    relations["identity_available_at_utc"] = "2020-01-01T00:00:00"
    with pytest.raises(DataReadinessError, match="aware UTC"):
        map_news_relations(relations, _bridge())


def test_share_classes_match_exact_ticker_even_with_same_cik() -> None:
    old = [_row(f"cik:0001652044:ticker:{ticker}", ticker) for ticker in ("GOOG", "GOOGL")]
    target = [_row(f"class:{ticker}", ticker) for ticker in ("GOOG", "GOOGL")]
    sec = [{**row, "sec_cik": "0001652044"} for row in target]
    inputs = [pd.DataFrame(rows) for rows in (old, old, target, sec)]
    bridge = build_issuer_news_bridge(*inputs, evidence_sha256=PIN)
    assert dict(zip(bridge.ticker, bridge.target_security_id, strict=True)) == {"GOOG": "class:GOOG", "GOOGL": "class:GOOGL"}
    shuffled = [frame.iloc[::-1].reset_index(drop=True) for frame in inputs]
    pd.testing.assert_frame_equal(bridge, build_issuer_news_bridge(*shuffled, evidence_sha256=PIN))
    inputs[2] = inputs[2].iloc[:1]
    assert list(build_issuer_news_bridge(*inputs, evidence_sha256=PIN).ticker) == ["GOOG"]


@pytest.mark.parametrize("security", ["historical:" + "b" * 64, "cusip:037833100", "hash:abcdef"])
def test_unproved_hash_and_cusip_source_are_not_guessed(security: str) -> None:
    inputs = _inputs()
    inputs[0]["security_id"] = inputs[1]["security_id"] = security
    assert build_issuer_news_bridge(*inputs, evidence_sha256=PIN).empty


@pytest.mark.parametrize("frame_index", [0, 1, 2, 3])
def test_missing_authority_never_restores_source_exclusion(frame_index: int) -> None:
    inputs = _inputs()
    inputs[frame_index] = inputs[frame_index].iloc[:0]
    assert build_issuer_news_bridge(*inputs, evidence_sha256=PIN).empty


def test_excluded_fi_and_wrong_legacy_pair_cannot_be_restored() -> None:
    inputs = _inputs()
    inputs[0] = pd.DataFrame([_row("cik:0000798354:ticker:FI", "FI")])
    inputs[2] = pd.DataFrame([_row("cik:0000798354", "FI")])
    inputs[3] = pd.DataFrame([_row("cik:0000798354", "FI", sec="0000798354")])
    for row in (_row("cik:0000798354", "FI"), _row("cik:0000798354:ticker:FISV", "FISV")):
        inputs[1] = pd.DataFrame([row])
        assert build_issuer_news_bridge(*inputs, evidence_sha256=PIN).empty


@pytest.mark.parametrize("frame_index", [0, 1, 2, 3])
def test_duplicate_and_overlapping_authorities_fail(frame_index: int) -> None:
    for start in (1, 2):
        inputs = _inputs()
        extra = inputs[frame_index].copy()
        extra.loc[0, "effective_from_utc"] = _time(start)
        inputs[frame_index] = pd.concat([inputs[frame_index], extra], ignore_index=True)
        with pytest.raises(DataReadinessError, match="overlapping"):
            build_issuer_news_bridge(*inputs, evidence_sha256=PIN)


@pytest.mark.parametrize("authority", [2, 3])
def test_global_conflict_rejected_before_noop_or_source_filter(authority: int) -> None:
    inputs = _inputs()
    inputs[0]["security_id"] = inputs[1]["security_id"] = NEW
    extra = inputs[authority].copy()
    extra["security_id"] = "another-target"
    inputs[authority] = pd.concat([inputs[authority], extra], ignore_index=True)
    with pytest.raises(DataReadinessError, match="overlapping"):
        build_issuer_news_bridge(*inputs, evidence_sha256=PIN)


def test_ticker_reuse_and_adjacent_authority_boundaries() -> None:
    inputs = _inputs()
    inputs[2] = pd.DataFrame([_row(NEW, end=10), _row("cik:0000000002", start=10)])
    inputs[3] = pd.DataFrame([_row(NEW, end=10, sec="0000320193"),
        _row("cik:0000000002", start=10, sec="0000000002")])
    bridge = build_issuer_news_bridge(*inputs, evidence_sha256=PIN)
    assert len(bridge) == 1
    assert bridge.iloc[0].effective_to_utc == _time(10)
    mapped = map_news_relations(_relations(), bridge)
    assert list(mapped.identity_translation_status) == ["mapped", "mapped", "unmapped", "unmapped"]
    inputs[0]["effective_from_utc"] = _time(10)
    assert build_issuer_news_bridge(*inputs, evidence_sha256=PIN).empty


@pytest.mark.parametrize(("column", "value"), [
    ("security_id", "cik:320193"), ("security_id", "cik:0000320193:ticker:MSFT"),
    ("security_id", "cik:0000000000"), ("security_id", " whitespace"), ("ticker", "aapl"),
    ("ticker", "AA/PL"), ("effective_from_utc", "2020-01-01"),
    ("effective_from_utc", "2020-01-01T01:00:00+01:00"), ("effective_from_utc", float("inf")),
    ("effective_from_utc", pd.NaT), ("effective_to_utc", "NaT"),
    ("effective_to_utc", "2020-01-01T00:00:00Z"), ("available_at_utc", None),
])
def test_noncanonical_inputs_fail(column: str, value: object) -> None:
    inputs = _inputs()
    inputs[0][column] = pd.Series([value], dtype=object)
    with pytest.raises(DataReadinessError):
        build_issuer_news_bridge(*inputs, evidence_sha256=PIN)


@pytest.mark.parametrize("proof", ["320193", 320193, "0000000000", "0000000002"])
def test_sec_cik_requires_canonical_and_consistent_proof(proof: object) -> None:
    inputs = _inputs()
    inputs[3]["sec_cik"] = proof
    with pytest.raises(DataReadinessError):
        build_issuer_news_bridge(*inputs, evidence_sha256=PIN)


def test_same_ids_are_not_bridge_conversions() -> None:
    inputs = _inputs()
    inputs[0]["security_id"] = inputs[1]["security_id"] = NEW
    assert build_issuer_news_bridge(*inputs, evidence_sha256=PIN).empty


def test_relations_preserve_source_scores_and_provenance_and_use_event_clock() -> None:
    inputs = _inputs()
    inputs[1]["available_at_utc"] = _time(5)
    inputs[3]["effective_to_utc"] = _time(15)
    bridge = build_issuer_news_bridge(*inputs, evidence_sha256=PIN)
    relations = _relations()
    relations.index = [9, 9, 5, 1]
    before = relations.copy(deep=True)
    result = map_news_relations(relations, bridge)
    assert list(result.identity_translation_status) == ["unmapped", "mapped", "mapped", "unmapped"]
    assert list(result.target_security_id) == [OLD, NEW, NEW, OLD]
    changed = {"target_security_id", "relation_id", "identity_available_at_utc", "feature_available_at_utc"}
    for column in set(relations.columns) - changed:
        pd.testing.assert_series_equal(result[column], before[column])
    assert result.iloc[1].identity_original_relation_id == before.iloc[1].relation_id
    assert result.iloc[1].identity_original_target_security_id == OLD
    assert result.iloc[1].identity_available_at_utc == _time(5)
    assert result.iloc[1].feature_available_at_utc == _time(20)
    assert result.iloc[1].relation_id != before.iloc[1].relation_id
    assert result.iloc[0].relation_id == before.iloc[0].relation_id
    pd.testing.assert_frame_equal(relations, before)
    pd.testing.assert_frame_equal(result, map_news_relations(relations, bridge))


def test_future_identity_is_not_backdated_and_null_identity_is_supported() -> None:
    relations = _relations().iloc[:1].copy()
    relations["identity_available_at_utc"] = _time(2)
    assert map_news_relations(relations, _bridge()).iloc[0].identity_translation_status == "unmapped"
    relations["identity_available_at_utc"] = pd.Series(pd.NaT, index=relations.index, dtype="datetime64[ns, UTC]")
    result = map_news_relations(relations, _bridge())
    assert result.iloc[0].identity_available_at_utc == _time(1)


def test_relation_id_binds_original_and_bridge() -> None:
    relations = _relations()
    first = map_news_relations(relations, _bridge())
    bridge = build_issuer_news_bridge(*_inputs(), evidence_sha256="b" * 64)
    second = map_news_relations(relations, bridge)
    assert (first.relation_id != second.relation_id).all()
    assert first.relation_id.is_unique


@pytest.mark.parametrize("status", ["observed", "observed_empty", "unavailable_saved_evidence"])
def test_coverage_splits_availability_intervals_without_losing_blindspots(status: str) -> None:
    inputs = _inputs()
    inputs[3] = pd.DataFrame([_row(NEW, start=3, end=10, available=5, sec="0000320193"),
        _row(NEW, start=12, end=15, sec="0000320193")])
    bridge = build_issuer_news_bridge(*inputs, evidence_sha256=PIN)
    ledger = _ledger().assign(status=status)
    before = ledger.copy(deep=True)
    result = map_news_coverage(ledger, bridge)
    assert list(result.requested_start_utc) == [_time(day) for day in (1, 5, 10, 12, 15)]
    assert list(result.requested_end_utc) == [_time(day) for day in (5, 10, 12, 15, 20)]
    assert list(result.security_id) == [OLD, NEW, OLD, NEW, OLD]
    assert result.status.eq(status).all() and result.coverage_blindspot.all()
    assert result.row_count.eq(0).all() and result.chunk_id.is_unique
    assert result.identity_original_chunk_id.eq("old-chunk").all()
    assert result.identity_original_security_id.eq(OLD).all()
    assert result.identity_original_requested_start_utc.eq(_time(1)).all()
    assert result.identity_original_requested_end_utc.eq(_time(20)).all()
    assert result.preparation_original_chunk_id.eq("physical-child").all()
    assert result.preparation_source_child_set_sha256.eq(PIN).all()
    assert sum(result.requested_end_utc - result.requested_start_utc, pd.Timedelta(0)) == _time(20) - _time(1)
    pd.testing.assert_frame_equal(ledger, before)
    pd.testing.assert_frame_equal(result, map_news_coverage(ledger, bridge))


def test_coverage_never_extends_original_interval_or_uses_future_proof() -> None:
    ledger = _ledger()
    result = map_news_coverage(ledger, _bridge())
    assert list(result.requested_start_utc) == list(ledger.requested_start_utc)
    assert list(result.requested_end_utc) == list(ledger.requested_end_utc)
    assert result.iloc[0].security_id == NEW
    assert result.iloc[0].chunk_id != "old-chunk"
    inputs = _inputs()
    inputs[1]["available_at_utc"] = _time(20)
    future = map_news_coverage(ledger, build_issuer_news_bridge(*inputs, evidence_sha256=PIN))
    assert future.iloc[0].identity_translation_status == "unmapped"
    assert future.iloc[0].chunk_id == "old-chunk"
    assert future.iloc[0].security_id == OLD


@pytest.mark.parametrize(("function", "fixture"), [(map_news_relations, _relations), (map_news_coverage, _ledger)])
def test_unmapped_and_empty_inputs_have_provenance(
    function: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame], fixture: Callable[[], pd.DataFrame],
) -> None:
    empty_bridge = _bridge().iloc[:0]
    frame = fixture()
    result = function(frame, empty_bridge)
    for column in frame.columns:
        pd.testing.assert_series_equal(result[column], frame[column])
    assert result.identity_translation_status.eq("unmapped").all()
    empty = function(frame.iloc[:0], empty_bridge)
    assert empty.empty and list(empty.columns) == list(result.columns)
    if function is map_news_coverage:
        assert {"identity_original_security_id", "identity_original_chunk_id"} <= set(empty.columns)


@pytest.mark.parametrize(("function", "fixture"), [(map_news_relations, _relations), (map_news_coverage, _ledger)])
def test_mappers_reject_ambiguous_bridge_and_repeated_translation(
    function: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame], fixture: Callable[[], pd.DataFrame],
) -> None:
    bridge = _bridge()
    with pytest.raises(DataReadinessError, match="overlapping"):
        function(fixture(), pd.concat([bridge, bridge], ignore_index=True))
    with pytest.raises(DataReadinessError, match="untranslated"):
        function(function(fixture(), bridge), bridge)
    with pytest.raises(DataReadinessError, match="unique"):
        function(pd.concat([fixture(), fixture()], ignore_index=True), bridge)


def test_invalid_pins_and_bounded_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(DataReadinessError, match="SHA256"):
        build_issuer_news_bridge(*_inputs(), evidence_sha256="A" * 64)
    bridge = _bridge()
    monkeypatch.setattr(identity, "MAXIMUM_ROWS", 1)
    with pytest.raises(DataReadinessError, match="bounded"):
        map_news_relations(_relations(), bridge)
    monkeypatch.setattr(identity, "MAXIMUM_FRAME_BYTES", 1)
    with pytest.raises(DataReadinessError, match="bounded"):
        build_issuer_news_bridge(*_inputs(), evidence_sha256=PIN)


def test_coverage_output_expansion_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge, ledger = _bridge(), _ledger()
    limit = max(int(frame.memory_usage(deep=True).sum()) for frame in (bridge, ledger))
    monkeypatch.setattr(identity, "MAXIMUM_FRAME_BYTES", limit)
    with pytest.raises(DataReadinessError, match="bounded"):
        map_news_coverage(ledger, bridge)


def test_bare_cik_source_still_requires_exact_ticker_and_sec_proof() -> None:
    inputs = _inputs()
    inputs[0]["security_id"] = inputs[1]["security_id"] = NEW
    inputs[2]["security_id"] = inputs[3]["security_id"] = "class:AAPL"
    bridge = build_issuer_news_bridge(*inputs, evidence_sha256=PIN)
    assert list(bridge.source_security_id) == [NEW]
    assert list(bridge.target_security_id) == ["class:AAPL"]
    inputs[3]["ticker"] = "MSFT"
    assert build_issuer_news_bridge(*inputs, evidence_sha256=PIN).empty


def test_relation_lookup_uses_target_not_source_identity() -> None:
    relations = _relations()
    relations["target_security_id"] = "unproved:target"
    result = map_news_relations(relations, _bridge())
    assert result.identity_translation_status.eq("unmapped").all()
    assert result.target_security_id.eq("unproved:target").all()


def test_coverage_chunk_hash_binds_original_interval_and_bridge() -> None:
    first = map_news_coverage(_ledger(), _bridge()).iloc[0].chunk_id
    changed_original = map_news_coverage(_ledger().assign(chunk_id="another-chunk"), _bridge()).iloc[0].chunk_id
    changed_interval = map_news_coverage(_ledger().assign(requested_end_utc=_time(19)), _bridge()).iloc[0].chunk_id
    other_bridge = build_issuer_news_bridge(*_inputs(), evidence_sha256="b" * 64)
    changed_proof = map_news_coverage(_ledger(), other_bridge).iloc[0].chunk_id
    assert len({first, changed_original, changed_interval, changed_proof}) == 4
