"""Disk-backed causal history exclusions with unchanged source checks.

No provider source database writes, model labels, thresholds or admission policy.
The caller supplies its existing clock/identity/resource implementations unchanged.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

_BATCH_SIZE = 256
_SCHEMA = """
CREATE TABLE links (
  source_family TEXT NOT NULL, event_id TEXT NOT NULL, kind TEXT NOT NULL,
  token TEXT NOT NULL, available_ns INTEGER NOT NULL, security_id TEXT NOT NULL,
  PRIMARY KEY(source_family,event_id,kind,token,available_ns,security_id)
) WITHOUT ROWID;
CREATE TABLE targets (
  version_id TEXT NOT NULL, security_id TEXT NOT NULL,
  PRIMARY KEY(version_id,security_id)
) WITHOUT ROWID;
CREATE TABLE ambiguity (
  version_id TEXT NOT NULL PRIMARY KEY, reason TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE exclusions (
  source_family TEXT NOT NULL, event_id TEXT NOT NULL, security_id TEXT NOT NULL, reason TEXT NOT NULL,
  PRIMARY KEY(source_family,event_id,security_id,reason)
) WITHOUT ROWID;
"""


@dataclass(frozen=True)
class HistoryExclusions:
    """A temporary index, valid only within staged_history_exclusions's context."""

    connection: sqlite3.Connection
    path: Path

    def owners(self, version_id: str) -> tuple[str, ...]:
        return tuple(row[0] for row in self.connection.execute(
            "SELECT security_id FROM targets WHERE version_id=? ORDER BY security_id", (version_id,),
        ))

    def ambiguous_reason(self, version_id: str) -> str | None:
        row = self.connection.execute("SELECT reason FROM ambiguity WHERE version_id=?", (version_id,)).fetchone()
        return None if row is None else str(row[0])

    def history_errors(self, source_family: str, event_id: str, version_id: str) -> list[str]:
        return [row[0] for row in self.connection.execute(
            "SELECT DISTINCT e.reason FROM targets AS t JOIN exclusions AS e ON e.security_id=t.security_id "
            "WHERE t.version_id=? AND e.source_family=? AND e.event_id=? ORDER BY e.reason",
            (version_id, source_family, event_id),
        )]

    def excluded_histories(self) -> Iterator[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT source_family,event_id,security_id,reason FROM exclusions "
            "ORDER BY source_family,event_id,security_id,reason",
        )
        for (source, event, security), grouped in groupby(rows, key=lambda row: (row[0], row[1], row[2])):
            yield {"source_family": source, "source_id": event, "security_id": security,
                   "reasons": [row[3] for row in grouped]}


def _exact_owners(
    stage: sqlite3.Connection, source: str, event: str, cluster: str, query_id: Any, cutoff_ns: int | None,
) -> tuple[str, ...]:
    bound = "" if cutoff_ns is None else " AND available_ns<=?"
    first: tuple[Any, ...] = (source, event, cluster)
    # Preserve the old typed dictionary lookup. SQLite TEXT affinity would
    # otherwise equate a malformed integer query ID with an issuer string.
    query_token = query_id if isinstance(query_id, str) and query_id else None
    second: tuple[Any, ...] = (source, event, query_token)
    if cutoff_ns is not None:
        first += (cutoff_ns,)
        second += (cutoff_ns,)
    # UNION deduplicates issuers even when proofs appear at several clocks or in
    # both version and retained-query copies. NULL query_id matches no text token.
    return tuple(row[0] for row in stage.execute(
        "SELECT security_id FROM links WHERE source_family=? AND event_id=? AND kind='cluster' AND token=?" + bound
        + " UNION SELECT security_id FROM links WHERE source_family=? AND event_id=? AND kind='query' AND token=?" + bound
        + " ORDER BY security_id", first + second,
    ))


def _build(
    source_connection: sqlite3.Connection, stage: sqlite3.Connection, *, end: Any,
    clock: Callable[[Any], Any], identity_errors: Callable[[dict[str, Any]], list[str]],
    guard_batch: Callable[[int], None], guard: Callable[[], None],
) -> None:
    for query in ("SELECT cluster_id,content_json FROM versions",
                  "SELECT cluster_id,record_json FROM records WHERE version_id!=''"):
        for position, (cluster, encoded) in enumerate(source_connection.execute(query)):
            if position and position % _BATCH_SIZE == 0:
                stage.commit()
            guard_batch(position)
            row = json.loads(encoded)
            version_clock = clock(row.get("version_available_at_utc"))
            if version_clock is None or version_clock > end or identity_errors(row):
                continue
            identity_clock = clock(row["identity_available_at_utc"])
            assert identity_clock is not None
            available = max(version_clock, identity_clock)
            if available > end:
                continue
            security, source, event = row["security_id"], row["source_family"], row["source_id"]
            keys = [("cluster", cluster), ("query", security), ("event", "*")]
            metadata = row.get("source_metadata", {})
            query_id, resolution = metadata.get("query_security_id"), metadata.get("query_identity_resolution")
            proof = metadata.get("identity_bridge_row_sha256" if resolution == "bridged" else "identity_legacy_proof_row_sha256")
            proven = (resolution == "identity_equal" and query_id == security) or (
                resolution in ("bridged", "proven_legacy_identity") and metadata.get("cohort_security_id") == security
                and isinstance(proof, str) and len(proof) == 64 and set(proof) <= set("0123456789abcdef"))
            if proven and isinstance(query_id, str) and query_id:
                keys.append(("query", query_id))
            stage.executemany("INSERT OR IGNORE INTO links VALUES (?,?,?,?,?,?)",
                              ((source, event, kind, token, int(available.value), security) for kind, token in keys))
        stage.commit()
        guard()

    for position, (cluster, version, encoded) in enumerate(source_connection.execute(
        "SELECT cluster_id,version_id,content_json FROM versions",
    )):
        if position and position % _BATCH_SIZE == 0:
            stage.commit()
        guard_batch(position)
        row = json.loads(encoded)
        version_clock = clock(row.get("version_available_at_utc"))
        if version_clock is not None and version_clock > end:
            # Absence from targets means the original empty owner tuple. A
            # future identity failure must not suppress an in-period history.
            continue
        source, event, security = row["source_family"], row["source_id"], row.get("security_id")
        ambiguous = False
        owners: tuple[str, ...]
        if isinstance(security, str) and security:
            owners = (security,)
        else:
            query_id = row.get("source_metadata", {}).get("query_security_id")
            owners = (() if version_clock is None else _exact_owners(
                stage, source, event, cluster, query_id, int(version_clock.value),
            ))
            if len(owners) != 1:
                ambiguous = True
                stage.execute("INSERT INTO ambiguity VALUES (?,?)", (version, "ambiguous_source_event_issuer_lineage"))
                if not owners:
                    # Later exact links suppress histories, without assigning
                    # identity backward. Event-wide fallback is last resort.
                    owners = _exact_owners(stage, source, event, cluster, query_id, None)
                    if not owners:
                        owners = tuple(row[0] for row in stage.execute(
                            "SELECT DISTINCT security_id FROM links WHERE source_family=? AND event_id=? "
                            "AND kind='event' AND token='*' ORDER BY security_id", (source, event),
                        ))
        stage.executemany("INSERT INTO targets VALUES (?,?)", ((version, owner) for owner in owners))
        errors = identity_errors(row)
        if ambiguous:
            errors.append("ambiguous_source_event_issuer_lineage")
        stage.executemany("INSERT OR IGNORE INTO exclusions VALUES (?,?,?,?)",
                          ((source, event, owner, reason) for owner in owners for reason in errors))
    stage.commit()
    guard()


@contextmanager
def staged_history_exclusions(
    source_connection: sqlite3.Connection, *, directory: Path, end: Any,
    clock: Callable[[Any], Any], identity_errors: Callable[[dict[str, Any]], list[str]],
    guard_batch: Callable[[int], None], guard: Callable[[], None],
) -> Iterator[HistoryExclusions]:
    """Never modify/attach to the pinned source; clean private staging on failure."""
    guard()
    with TemporaryDirectory(prefix="issuer-history-", dir=directory) as temporary:
        path = Path(temporary) / "history.sqlite"
        stage = sqlite3.connect(path)
        try:
            stage.execute("PRAGMA journal_mode=DELETE")
            stage.execute("PRAGMA cache_size=-8192")
            stage.execute("PRAGMA temp_store=FILE")
            stage.execute("PRAGMA mmap_size=0")
            stage.executescript(_SCHEMA)
            _build(source_connection, stage, end=end, clock=clock, identity_errors=identity_errors,
                   guard_batch=guard_batch, guard=guard)
            yield HistoryExclusions(stage, path)
            guard()
        finally:
            stage.close()
