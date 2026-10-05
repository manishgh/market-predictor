"""Synthetic unit cases for causal history ownership and immutable-source cleanup."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from market_predictor.core.errors import MemoryBudgetError
from market_predictor.research.issuer_content_history import staged_history_exclusions
from market_predictor.research.issuer_content_qualification_authority import END, _clock, _identity_errors


def _row(security: str | None, clock: str) -> dict[str, Any]:
    return {"source_family": "alpaca", "source_id": "story", "security_id": security,
            "ticker": None if security is None else security.upper(), "source_version_sha256": "a" * 64,
            "published_at_utc": clock, "version_available_at_utc": clock, "event_available_at_utc": clock,
            "identity_available_at_utc": "2020-01-01T00:00:00Z", "identity_authority_sha256": "b" * 64,
            "source_metadata": {}, "unavailable_reasons": []}


@contextmanager
def _source(
    path: Path, versions: list[tuple[str, str, dict[str, Any]]],
) -> Iterator[tuple[sqlite3.Connection, str]]:
    with sqlite3.connect(path) as builder:
        builder.executescript("CREATE TABLE versions(version_id TEXT PRIMARY KEY,cluster_id TEXT,content_json TEXT); "
                             "CREATE TABLE records(cluster_id TEXT,version_id TEXT,record_json TEXT);")
        builder.executemany("INSERT INTO versions VALUES (?,?,?)",
                            ((version, cluster, json.dumps(row, sort_keys=True)) for version, cluster, row in versions))
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    reader = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    reader.execute("PRAGMA query_only=ON")
    try:
        yield reader, original
        assert reader.execute("PRAGMA query_only").fetchone() == (1,)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == original
    finally:
        reader.close()


def _settings(tmp_path: Path) -> dict[str, Any]:
    return dict(directory=tmp_path, end=END, clock=_clock, identity_errors=_identity_errors,
                guard_batch=lambda position: None, guard=lambda: None)


def test_multiple_causal_owners_do_not_add_later_exact_owner(tmp_path: Path) -> None:
    versions = [("a", "shared", _row("issuer-a", "2020-06-01T20:00:00Z")),
                ("b", "shared", _row("issuer-b", "2020-06-01T20:00:00Z")),
                ("late", "shared", _row("issuer-late", "2020-06-03T20:00:00Z")),
                ("unknown", "shared", _row(None, "2020-06-02T20:00:00Z"))]
    with _source(tmp_path / "source.sqlite", versions) as (source, _):
        with staged_history_exclusions(source, **_settings(tmp_path)) as history:
            stage_path = history.path
            assert history.owners("unknown") == ("issuer-a", "issuer-b")
            assert history.ambiguous_reason("unknown") == "ambiguous_source_event_issuer_lineage"
            reasons = ["ambiguous_source_event_issuer_lineage", "unproven_source_identity"]
            assert history.history_errors("alpaca", "story", "a") == reasons
            assert history.history_errors("alpaca", "story", "b") == reasons
            assert history.history_errors("alpaca", "story", "late") == []
            assert list(history.excluded_histories()) == [
                {"source_family": "alpaca", "source_id": "story", "security_id": security, "reasons": reasons}
                for security in ("issuer-a", "issuer-b")]
            assert history.connection.execute("PRAGMA cache_size").fetchone() == (-8192,)
            assert history.connection.execute("PRAGMA temp_store").fetchone() == (1,)
            assert history.connection.execute("PRAGMA mmap_size").fetchone() == (0,)
        assert not stage_path.exists()


def test_later_exact_owner_precedes_unrelated_event_wide_fallback(tmp_path: Path) -> None:
    versions = [("exact-late", "exact", _row("issuer-exact", "2020-06-03T20:00:00Z")),
                ("unrelated", "other", _row("issuer-unrelated", "2020-06-01T20:00:00Z")),
                ("unknown", "exact", _row(None, "2020-06-02T20:00:00Z")),
                ("future", "exact", _row(None, "2026-01-01T20:00:00Z"))]
    with _source(tmp_path / "source.sqlite", versions) as (source, _):
        with staged_history_exclusions(source, **_settings(tmp_path)) as history:
            assert history.owners("unknown") == ("issuer-exact",)
            assert history.ambiguous_reason("unknown") == "ambiguous_source_event_issuer_lineage"
            assert history.history_errors("alpaca", "story", "unrelated") == []
            assert history.owners("future") == () and history.ambiguous_reason("future") is None
            assert [item["security_id"] for item in history.excluded_histories()] == ["issuer-exact"]


def test_numeric_query_id_does_not_match_string_issuer(tmp_path: Path) -> None:
    unknown = _row(None, "2020-06-02T20:00:00Z")
    unknown["source_metadata"] = {"query_security_id": 123}
    versions = [("proof", "proof", _row("123", "2020-06-01T20:00:00Z")),
                ("unknown", "other", unknown)]
    with _source(tmp_path / "source.sqlite", versions) as (source, _):
        with staged_history_exclusions(source, **_settings(tmp_path)) as history:
            assert history.owners("unknown") == ("123",)
            assert history.ambiguous_reason("unknown") == "ambiguous_source_event_issuer_lineage"
            assert history.history_errors("alpaca", "story", "proof") == [
                "ambiguous_source_event_issuer_lineage", "unproven_source_identity"]


@pytest.mark.parametrize("version_clock,ambiguous", [
    ("2020-06-01T20:00:00.000000001Z", True),
    ("2020-06-01T20:00:00.000000002Z", False),
    ("not-a-clock", True),
])
def test_clock_nanoseconds_and_missing_clock_preserve_causal_fallback(
    tmp_path: Path, version_clock: str, ambiguous: bool,
) -> None:
    versions = [("proof", "exact", _row("issuer-a", "2020-06-01T20:00:00.000000002Z")),
                ("unknown", "exact", _row(None, version_clock))]
    with _source(tmp_path / "source.sqlite", versions) as (source, _):
        with staged_history_exclusions(source, **_settings(tmp_path)) as history:
            assert history.owners("unknown") == ("issuer-a",)
            assert (history.ambiguous_reason("unknown") is not None) is ambiguous
            errors = history.history_errors("alpaca", "story", "proof")
            assert ("unproven_source_clock" in errors) is (version_clock == "not-a-clock")
            assert ("ambiguous_source_event_issuer_lineage" in errors) is ambiguous


@pytest.mark.parametrize("failure_phase", ["links", "targets", "consumer"])
def test_resource_failure_cleans_stage_and_keeps_source_immutable(tmp_path: Path, failure_phase: str) -> None:
    versions = [("proof", "exact", _row("issuer-a", "2020-06-01T20:00:00Z")),
                ("unknown", "exact", _row(None, "2020-06-02T20:00:00Z"))]
    stage_directory = tmp_path / "staging"
    stage_directory.mkdir()
    batches = 0

    def pressure(position: int) -> None:
        nonlocal batches
        if position == 0:
            batches += 1
        # The empty retained-record table has no per-row guard, so batch entry1
        # is version-link construction and entry2 is version-target construction.
        if (failure_phase == "links" and batches == 1) or (failure_phase == "targets" and batches == 2):
            raise MemoryBudgetError("synthetic history staging resource pressure")

    with _source(tmp_path / "source.sqlite", versions) as (source, original):
        settings = _settings(stage_directory) | {"guard_batch": pressure}
        with pytest.raises(MemoryBudgetError, match="resource pressure"):
            with staged_history_exclusions(source, **settings):
                assert failure_phase == "consumer"
                raise MemoryBudgetError("synthetic consumer resource pressure")
        assert list(stage_directory.iterdir()) == []
        assert hashlib.sha256((tmp_path / "source.sqlite").read_bytes()).hexdigest() == original
