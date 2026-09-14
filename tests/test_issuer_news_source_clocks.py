"""Recover compact timezone loss from exact pinned source relations, never guesses."""
from pathlib import Path

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.datasets.initial_fit_issuer_news import _write_frame
from market_predictor.swing.datasets.issuer_news_source_clocks import restore_compact_identity_clocks


def _inputs(tmp_path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, str]], Path]:
    original = pd.DataFrame({"relation_id": ["r1", "r2"], "event_id": ["e1", "e2"],
        "target_security_id": ["source", "source"], "target_ticker": ["ABC", "ABC"],
        "identity_available_at_utc": pd.to_datetime(["2019-07-09T04:00:00Z", None], utc=True)})
    path = tmp_path / "attribution/relations/chunk.parquet"
    record = _write_frame(original, path, "event_security_relations", {"test": "synthetic"})
    sidecar = path.with_name(path.name + ".manifest.json")
    pins = {str(path): record["sha256"], str(sidecar): file_sha256(sidecar)}
    compact = original.copy()
    compact["identity_available_at_utc"] = compact.identity_available_at_utc.dt.tz_localize(None).dt.as_unit("ms")
    compact["preparation_original_chunk_id"] = "chunk"
    compact["preparation_source_child_set_sha256"] = json_sha256(pins)
    return compact, {"chunk": pins}, path


def test_restore_lossy_clock_from_exact_source_preserves_nulls_and_provenance(tmp_path: Path) -> None:
    compact, children, _ = _inputs(tmp_path)
    original = compact.copy(deep=True)
    result = restore_compact_identity_clocks(compact, children)
    assert result.identity_available_at_utc.iloc[0] == pd.Timestamp("2019-07-09T04:00:00Z")
    assert pd.isna(result.identity_available_at_utc.iloc[1])
    pd.testing.assert_series_equal(result.compact_original_identity_available_at_utc,
        original.identity_available_at_utc, check_names=False)
    pd.testing.assert_frame_equal(compact, original)


@pytest.mark.parametrize("mutation", ["clock", "relation", "event", "security", "ticker", "proof", "missing"])
def test_restore_rejects_unproved_compact_values(tmp_path: Path, mutation: str) -> None:
    compact, children, _ = _inputs(tmp_path)
    if mutation == "clock":
        compact.loc[0, "identity_available_at_utc"] += pd.Timedelta(seconds=1)
    elif mutation == "missing":
        children = {}
    else:
        column = {"relation": "relation_id", "event": "event_id", "security": "target_security_id",
            "ticker": "target_ticker", "proof": "preparation_source_child_set_sha256"}[mutation]
        compact.loc[0, column] = "wrong"
    with pytest.raises(DataReadinessError):
        restore_compact_identity_clocks(compact, children)


def test_restore_rejects_changed_original_file(tmp_path: Path) -> None:
    compact, children, path = _inputs(tmp_path)
    path.write_bytes(b"changed")
    with pytest.raises(DataReadinessError, match="pin"):
        restore_compact_identity_clocks(compact, children)


def test_restore_does_not_normalize_already_aware_or_entirely_null(tmp_path: Path) -> None:
    compact, _, _ = _inputs(tmp_path)
    for values in (pd.to_datetime(["2019-07-09T04:00:00Z", None], utc=True), pd.to_datetime([None, None])):
        compact["identity_available_at_utc"] = values
        pd.testing.assert_frame_equal(restore_compact_identity_clocks(compact, {}), compact)


@pytest.mark.parametrize("mutation", ["duplicate", "naive", "null", "submillisecond"])
def test_restore_rejects_invalid_original_clock_proof(tmp_path: Path, mutation: str) -> None:
    compact, _, path = _inputs(tmp_path)
    original = pd.read_parquet(path)
    if mutation == "duplicate":
        original = pd.concat([original, original.iloc[:1]], ignore_index=True)
    elif mutation == "naive":
        original["identity_available_at_utc"] = original.identity_available_at_utc.dt.tz_localize(None)
    elif mutation == "null":
        original.loc[0, "identity_available_at_utc"] = pd.NaT
    else:
        original["identity_available_at_utc"] = original.identity_available_at_utc.dt.as_unit("ns")
        original.loc[0, "identity_available_at_utc"] += pd.Timedelta(nanoseconds=1)
    path = tmp_path / "modified/attribution/relations/chunk.parquet"
    record = _write_frame(original, path, "event_security_relations", {"test": "synthetic"})
    pins = {str(path): record["sha256"], str(path.with_name(path.name + ".manifest.json")):
        file_sha256(path.with_name(path.name + ".manifest.json"))}
    compact["preparation_source_child_set_sha256"] = json_sha256(pins)
    with pytest.raises(DataReadinessError):
        restore_compact_identity_clocks(compact, {"chunk": pins})


def test_restore_persisted_timezone_loss_and_preserve_naive_provenance(tmp_path: Path) -> None:
    compact, children, _ = _inputs(tmp_path)
    path = tmp_path / "compact.parquet"
    _write_frame(compact, path, "event_security_relations", {"test": "synthetic"})
    loaded, _ = load_canonical_artifact(path, expected_type="event_security_relations", allow_research=True)
    restored = restore_compact_identity_clocks(loaded, children)
    target = tmp_path / "restored.parquet"
    _write_frame(restored, target, "event_security_relations", {"test": "synthetic"})
    result, _ = load_canonical_artifact(target, expected_type="event_security_relations", allow_research=True)
    assert result.identity_available_at_utc.iloc[0] == pd.Timestamp("2019-07-09T04:00:00Z")
    pd.testing.assert_series_equal(result.compact_original_identity_available_at_utc,
        compact.identity_available_at_utc, check_names=False)


def test_restoration_cannot_normalize_an_unvalidated_non_utc_row(tmp_path: Path) -> None:
    compact, children, _ = _inputs(tmp_path)
    compact["identity_available_at_utc"] = pd.Series([
        pd.Timestamp("2019-07-09T04:00:00"), pd.Timestamp("2019-07-09T05:00:00+01:00")], dtype=object)
    with pytest.raises(DataReadinessError, match="aware UTC"):
        restore_compact_identity_clocks(compact, children)
