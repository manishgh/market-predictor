"""Offline, hash-bound S&P security-transition authority.

The official S&P event artifact describes index additions and deletions.  It is
not, by itself, sufficient to follow a security through ticker changes.  This
module publishes the independent transition layer used by point-in-time
membership reconstruction.  It consumes only verified local evidence and never
performs network I/O.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, cast
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup

from market_predictor.canonical.store import file_sha256
from market_predictor.locking import LockTimeout, file_lock
from market_predictor.resources import assert_memory_budget, assert_peak_memory_budget
from market_predictor.v3.contracts import normalized_ticker
from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.spglobal_archive import (
    MAXIMUM_MEMORY_GIB,
    MEMORY_HEADROOM_GIB,
    VerifiedSpGlobalRawArchive,
    read_verified_spglobal_release_html,
    require_spglobal_raw_archive_complete,
)
from market_predictor.v3.spglobal_events import (
    require_spglobal_event_reconstruction_ready,
)

TRANSITION_REQUEST_SCHEMA: Final = "edge_rebuild.sp500_transition_request.v1"
TRANSITION_MANIFEST_SCHEMA: Final = "edge_rebuild.sp500_transition_manifest.v1"
TRANSITION_AUTHORITY_SCHEMA: Final = "edge_rebuild.sp500_transition_authority.v1"
TRANSITION_TABLE_SCHEMA: Final = "edge_rebuild.sp500_transitions.v1"
TRANSITION_FILE: Final = "transitions.parquet"

_REVIEWED_COLUMNS: Final = {
    "id",
    "effective_date",
    "old_symbol",
    "new_symbol",
    "transition_type",
    "identity_continuity",
    "membership_continuity",
    "old_security_id",
    "new_security_id",
    "source_url",
    "evidence_summary",
    "reviewed_by",
    "reviewed_at_utc",
}
_TRANSITION_COLUMNS: Final = (
    "transition_id",
    "effective_at_utc",
    "old_ticker",
    "new_ticker",
    "transition_type",
    "identity_continuity",
    "membership_continuity",
    "old_security_id",
    "new_security_id",
    "source_kind",
    "source_url",
    "source_sha256",
    "source_published_date",
    "atomic_group_id",
    "evidence_summary",
)
_EFFECTIVE_SENTENCE = re.compile(
    r"(?:also\s+)?effective\s+on\s+"
    r"(?P<effective>[A-Z][a-z]+\s+\d{1,2}(?:,\s*\d{4})?)\s*,\s*"
    r"(?P<body>.+?\bwill\s+change\s+its\b.+)",
    re.IGNORECASE,
)
_TICKER_CHANGE = re.compile(
    r"ticker\s+from\s+(?P<old>[A-Z0-9.-]+)\s+to\s+(?P<new>[A-Z0-9.-]+)",
    re.IGNORECASE,
)


def publish_sp500_transition_authority(
    *,
    archive_directory: Path,
    event_directory: Path,
    reviewed_transitions_path: Path,
    start_date: date,
    cutoff_date: date,
    output_directory: Path,
) -> dict[str, Any]:
    """Publish immutable transition evidence from verified local parents."""

    _validate_window(start_date, cutoff_date)
    output = output_directory.resolve()
    for parent in (archive_directory.resolve(), event_directory.resolve()):
        if output == parent or output in parent.parents or parent in output.parents:
            raise DataReadinessError("transition output and parent directories must be disjoint")
    output_directory.mkdir(parents=True, exist_ok=True)
    try:
        with file_lock(output_directory / "_publisher", timeout=0.0):
            return _publish_locked(
                archive_directory=archive_directory,
                event_directory=event_directory,
                reviewed_transitions_path=reviewed_transitions_path,
                start_date=start_date,
                cutoff_date=cutoff_date,
                output_directory=output_directory,
            )
    except LockTimeout as exc:
        raise DataReadinessError(f"another process is publishing S&P transitions {output_directory}") from exc


def require_sp500_transition_authority(
    transition_directory: Path,
    *,
    archive_directory: Path,
    event_directory: Path,
    reviewed_transitions_path: Path,
    start_date: date,
    cutoff_date: date,
) -> pd.DataFrame:
    """Replay parent evidence and return a verified transition table."""

    _validate_window(start_date, cutoff_date)
    archive, parent = _verified_parents(
        archive_directory=archive_directory,
        event_directory=event_directory,
    )
    expected_request = _request_payload(
        parent=parent,
        reviewed_transitions_path=reviewed_transitions_path,
        start_date=start_date,
        cutoff_date=cutoff_date,
    )
    request = _load_object(transition_directory / "_request.json")
    if request != {**expected_request, "request_sha256": _json_sha256(expected_request)}:
        raise DataReadinessError("S&P transition request identity is invalid")

    authority = _load_object(transition_directory / "_authority.json")
    manifest_path = _resolve_inside(
        transition_directory,
        str(authority.get("artifact", "")),
    )
    if (
        authority.get("schema") != TRANSITION_AUTHORITY_SCHEMA
        or authority.get("state") != "transition_complete"
        or not manifest_path.is_file()
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
    ):
        raise DataReadinessError("S&P transition authority is invalid")
    manifest = _load_object(manifest_path)
    request_sha256 = _json_sha256(expected_request)
    if (
        manifest.get("schema") != TRANSITION_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or authority.get("request_sha256") != request_sha256
        or manifest.get("parent_lineage") != parent
        or authority.get("parent_lineage") != parent
    ):
        raise DataReadinessError("S&P transition parent lineage is invalid")
    record = manifest.get("transition_artifact")
    if not isinstance(record, dict):
        raise DataReadinessError("S&P transition artifact record is invalid")
    transition_path = _resolve_inside(
        transition_directory,
        str(record.get("path", "")),
    )
    if (
        not transition_path.is_file()
        or record.get("sha256") != file_sha256(transition_path)
        or int(record.get("bytes", -1)) != transition_path.stat().st_size
    ):
        raise DataReadinessError("S&P transition artifact hash is invalid")
    actual = _read_transition_table(transition_path)
    expected = _build_transition_table(
        archive=archive,
        reviewed_transitions_path=reviewed_transitions_path,
        start_date=start_date,
        cutoff_date=cutoff_date,
    )
    if _records(actual) != _records(expected):
        raise DataReadinessError("S&P transition artifact does not replay from its parents")
    transition_set_sha256 = _records_sha256(actual)
    if (
        int(manifest.get("transition_count", -1)) != len(actual)
        or manifest.get("transition_set_sha256") != transition_set_sha256
        or authority.get("transition_set_sha256") != transition_set_sha256
    ):
        raise DataReadinessError("S&P transition counts or semantic identity are invalid")
    return actual


def _publish_locked(
    *,
    archive_directory: Path,
    event_directory: Path,
    reviewed_transitions_path: Path,
    start_date: date,
    cutoff_date: date,
    output_directory: Path,
) -> dict[str, Any]:
    assert_memory_budget(
        hard_budget_gib=MAXIMUM_MEMORY_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage="S&P transition publication start",
    )
    if (output_directory / "_authority.json").exists():
        raise DataReadinessError("completed S&P transition authority is immutable")
    archive, parent = _verified_parents(
        archive_directory=archive_directory,
        event_directory=event_directory,
    )
    request_payload = _request_payload(
        parent=parent,
        reviewed_transitions_path=reviewed_transitions_path,
        start_date=start_date,
        cutoff_date=cutoff_date,
    )
    request_sha256 = _json_sha256(request_payload)
    _write_json_atomic(
        output_directory / "_request.json",
        {**request_payload, "request_sha256": request_sha256},
    )
    transitions = _build_transition_table(
        archive=archive,
        reviewed_transitions_path=reviewed_transitions_path,
        start_date=start_date,
        cutoff_date=cutoff_date,
    )
    transition_path = output_directory / TRANSITION_FILE
    _write_parquet_atomic(transition_path, transitions)
    transition_set_sha256 = _records_sha256(transitions)
    source_counts = {str(key): int(value) for key, value in transitions["source_kind"].value_counts().sort_index().items()}
    manifest: dict[str, Any] = {
        "schema": TRANSITION_MANIFEST_SCHEMA,
        "status": "complete",
        "request_sha256": request_sha256,
        "parent_lineage": parent,
        "start_date": start_date.isoformat(),
        "cutoff_date": cutoff_date.isoformat(),
        "transition_count": len(transitions),
        "identity_continuous_count": int(transitions["identity_continuity"].sum()),
        "membership_continuous_count": int(transitions["membership_continuity"].sum()),
        "source_counts": source_counts,
        "transition_set_sha256": transition_set_sha256,
        "transition_artifact": _artifact_record(transition_path),
    }
    manifest_path = output_directory / "_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(output_directory / "_status.json", manifest)
    authority = {
        "schema": TRANSITION_AUTHORITY_SCHEMA,
        "state": "transition_complete",
        "artifact": "_manifest.json",
        "artifact_sha256": file_sha256(manifest_path),
        "request_sha256": request_sha256,
        "parent_lineage": parent,
        "transition_set_sha256": transition_set_sha256,
        "transition_count": len(transitions),
    }
    _write_json_atomic(output_directory / "_authority.json", authority)
    assert_peak_memory_budget(
        hard_budget_gib=MAXIMUM_MEMORY_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage="S&P transition publication",
    )
    require_sp500_transition_authority(
        output_directory,
        archive_directory=archive_directory,
        event_directory=event_directory,
        reviewed_transitions_path=reviewed_transitions_path,
        start_date=start_date,
        cutoff_date=cutoff_date,
    )
    return manifest


def _verified_parents(
    *,
    archive_directory: Path,
    event_directory: Path,
) -> tuple[VerifiedSpGlobalRawArchive, dict[str, str]]:
    archive = require_spglobal_raw_archive_complete(archive_directory)
    verified_events = require_spglobal_event_reconstruction_ready(
        event_directory,
        archive_directory=archive_directory,
    )
    return archive, {
        "raw_authority_sha256": file_sha256(archive.root / "_authority.json"),
        "raw_manifest_sha256": str(archive.authority["artifact_sha256"]),
        "raw_release_set_sha256": str(archive.manifest["release_set_sha256"]),
        "event_authority_sha256": verified_events.authority_sha256,
        "event_set_sha256": verified_events.event_set_sha256,
    }


def _request_payload(
    *,
    parent: Mapping[str, str],
    reviewed_transitions_path: Path,
    start_date: date,
    cutoff_date: date,
) -> dict[str, Any]:
    if not reviewed_transitions_path.is_file():
        raise DataReadinessError(f"reviewed transition ledger is missing: {reviewed_transitions_path}")
    return {
        "schema": TRANSITION_REQUEST_SCHEMA,
        "table_schema": TRANSITION_TABLE_SCHEMA,
        "start_date": start_date.isoformat(),
        "cutoff_date": cutoff_date.isoformat(),
        "reviewed_transition_sha256": file_sha256(reviewed_transitions_path),
        "parent_lineage": dict(parent),
    }


def _build_transition_table(
    *,
    archive: VerifiedSpGlobalRawArchive,
    reviewed_transitions_path: Path,
    start_date: date,
    cutoff_date: date,
) -> pd.DataFrame:
    rows = [
        *_official_transition_rows(
            archive,
            start_date=start_date,
            cutoff_date=cutoff_date,
        ),
        *_reviewed_transition_rows(
            reviewed_transitions_path,
            start_date=start_date,
            cutoff_date=cutoff_date,
        ),
    ]
    if not rows:
        raise DataReadinessError("S&P transition authority has no transition evidence")
    data = pd.DataFrame(rows, columns=_TRANSITION_COLUMNS)
    data["effective_at_utc"] = pd.to_datetime(data["effective_at_utc"], utc=True)
    data["identity_continuity"] = data["identity_continuity"].astype(bool)
    data["membership_continuity"] = data["membership_continuity"].astype(bool)
    data = _deduplicate_exact_transitions(data)
    _validate_transition_graph(data)
    return data.sort_values(
        ["effective_at_utc", "old_ticker", "new_ticker", "transition_id"],
        kind="stable",
    ).reset_index(drop=True)


def _official_transition_rows(
    archive: VerifiedSpGlobalRawArchive,
    *,
    start_date: date,
    cutoff_date: date,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for release in archive.releases:
        published = date.fromisoformat(str(release["published_date"]))
        html = read_verified_spglobal_release_html(archive, release)
        soup = BeautifulSoup(html, "html.parser")
        body = soup.select_one(".wd_news_body") or soup
        text = " ".join(body.get_text(" ", strip=True).split())
        sentences = re.split(r"(?<=[.!?])\s+(?=(?:Also\s+)?[A-Z])", text)
        for sentence in sentences:
            effective_match = _EFFECTIVE_SENTENCE.search(sentence)
            if effective_match is None:
                continue
            pairs = list(_TICKER_CHANGE.finditer(effective_match.group("body")))
            if not pairs:
                continue
            effective_date = _effective_date(
                effective_match.group("effective"),
                published=published,
            )
            if not start_date <= effective_date <= cutoff_date:
                continue
            effective_at = _session_midnight(effective_date)
            source_sha256 = str(release["sha256"])
            source_url = str(release["url"])
            group_id = _json_sha256(
                {
                    "effective_at_utc": effective_at.isoformat(),
                    "source_sha256": source_sha256,
                }
            )
            for pair in pairs:
                old_ticker = normalized_ticker(pair.group("old").rstrip("."))
                new_ticker = normalized_ticker(pair.group("new").rstrip("."))
                transition_id = _json_sha256(
                    {
                        "effective_at_utc": effective_at.isoformat(),
                        "old_ticker": old_ticker,
                        "new_ticker": new_ticker,
                        "source_sha256": source_sha256,
                    }
                )
                rows.append(
                    {
                        "transition_id": transition_id,
                        "effective_at_utc": effective_at,
                        "old_ticker": old_ticker,
                        "new_ticker": new_ticker,
                        "transition_type": "official_ticker_change",
                        "identity_continuity": True,
                        "membership_continuity": True,
                        "old_security_id": "",
                        "new_security_id": "",
                        "source_kind": "spglobal_official_release",
                        "source_url": source_url,
                        "source_sha256": source_sha256,
                        "source_published_date": published.isoformat(),
                        "atomic_group_id": group_id,
                        "evidence_summary": "Official release explicitly states the ticker transition.",
                    }
                )
    return rows


def _reviewed_transition_rows(
    path: Path,
    *,
    start_date: date,
    cutoff_date: date,
) -> list[dict[str, Any]]:
    reviewed = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(_REVIEWED_COLUMNS.difference(reviewed.columns))
    if missing:
        raise DataReadinessError(f"reviewed transition ledger is missing columns: {missing}")
    if reviewed.empty:
        raise DataReadinessError("reviewed transition ledger is empty")
    ledger_sha256 = file_sha256(path)
    rows: list[dict[str, Any]] = []
    for record in reviewed.to_dict(orient="records"):
        effective = _parse_date(str(record["effective_date"]), field="effective_date")
        if not start_date <= effective <= cutoff_date:
            continue
        old_ticker = normalized_ticker(str(record["old_symbol"]))
        new_ticker = normalized_ticker(str(record["new_symbol"]))
        transition_id = str(record["id"]).strip()
        if not transition_id:
            raise DataReadinessError("reviewed transition has an empty id")
        source_url = str(record["source_url"]).strip()
        if not source_url.startswith("https://"):
            raise DataReadinessError("reviewed transition source URL must use HTTPS")
        reviewed_at = pd.Timestamp(record["reviewed_at_utc"])
        if pd.isna(reviewed_at) or reviewed_at.tzinfo is None:
            raise DataReadinessError("reviewed transition has invalid reviewed_at_utc")
        rows.append(
            {
                "transition_id": transition_id,
                "effective_at_utc": _session_midnight(effective),
                "old_ticker": old_ticker,
                "new_ticker": new_ticker,
                "transition_type": str(record["transition_type"]).strip().lower(),
                "identity_continuity": _strict_bool(record["identity_continuity"]),
                "membership_continuity": _strict_bool(record["membership_continuity"]),
                "old_security_id": str(record["old_security_id"]).strip(),
                "new_security_id": str(record["new_security_id"]).strip(),
                "source_kind": "reviewed_transition_ledger",
                "source_url": source_url,
                "source_sha256": ledger_sha256,
                "source_published_date": effective.isoformat(),
                "atomic_group_id": transition_id,
                "evidence_summary": str(record["evidence_summary"]).strip(),
            }
        )
    return rows


def _deduplicate_exact_transitions(data: pd.DataFrame) -> pd.DataFrame:
    kept: list[pd.Series] = []
    for _, group in data.groupby(
        ["effective_at_utc", "old_ticker", "new_ticker"],
        sort=False,
        dropna=False,
    ):
        semantics = group[["identity_continuity", "membership_continuity"]].drop_duplicates()
        if len(semantics) != 1:
            raise DataReadinessError("conflicting duplicate S&P transition evidence")
        official = group[group["source_kind"].eq("spglobal_official_release")]
        kept.append((official if not official.empty else group).iloc[0])
    return pd.DataFrame(kept, columns=data.columns).reset_index(drop=True)


def _validate_transition_graph(data: pd.DataFrame) -> None:
    if bool(data["transition_id"].astype(str).str.strip().eq("").any()):
        raise DataReadinessError("S&P transition has an empty identity")
    if bool(data["transition_id"].duplicated().any()):
        raise DataReadinessError("S&P transition identities are duplicated")
    if bool(data["old_ticker"].eq(data["new_ticker"]).any()):
        raise DataReadinessError("S&P transition cannot map a ticker to itself")
    for effective_at, group in data.groupby("effective_at_utc", sort=True):
        duplicate_sources = group.groupby("old_ticker")["new_ticker"].nunique()
        if bool(duplicate_sources.gt(1).any()):
            tickers = sorted(duplicate_sources[duplicate_sources.gt(1)].index.astype(str))
            raise DataReadinessError(f"ambiguous S&P transition destinations at {effective_at}: {tickers}")
        duplicate_destinations = group.groupby("new_ticker")["old_ticker"].nunique()
        if bool(duplicate_destinations.gt(1).any()):
            tickers = sorted(duplicate_destinations[duplicate_destinations.gt(1)].index.astype(str))
            raise DataReadinessError(f"ambiguous S&P transition predecessors at {effective_at}: {tickers}")
        overlap = set(group["old_ticker"]).intersection(group["new_ticker"])
        if not overlap:
            continue
        # Simultaneous symbol swaps are valid only when one verified official
        # release states the complete atomic batch. Applying such rows
        # sequentially would corrupt the 2019 Fox identities.
        if (
            group["atomic_group_id"].nunique() != 1
            or group["source_kind"].nunique() != 1
            or group["source_kind"].iloc[0] != "spglobal_official_release"
            or group["source_sha256"].nunique() != 1
            or not bool(group["identity_continuity"].all())
        ):
            raise DataReadinessError(f"ambiguous same-time S&P transition chain at {effective_at}: {sorted(overlap)}")


def _read_transition_table(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path)
    missing = sorted(set(_TRANSITION_COLUMNS).difference(data.columns))
    if missing:
        raise DataReadinessError(f"S&P transition artifact is missing columns: {missing}")
    data = data.loc[:, list(_TRANSITION_COLUMNS)].copy()
    data["effective_at_utc"] = pd.to_datetime(data["effective_at_utc"], utc=True)
    _validate_transition_graph(data)
    return data.sort_values(
        ["effective_at_utc", "old_ticker", "new_ticker", "transition_id"],
        kind="stable",
    ).reset_index(drop=True)


def _effective_date(value: str, *, published: date) -> date:
    candidate = value if re.search(r"\d{4}", value) else f"{value}, {published.year}"
    parsed = _parse_date(candidate, field="official transition effective date")
    if parsed < published and (published - parsed).days > 180:
        parsed = parsed.replace(year=parsed.year + 1)
    return parsed


def _parse_date(value: str, *, field: str) -> date:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError(f"invalid {field}: {value!r}") from exc
    if pd.isna(parsed):
        raise DataReadinessError(f"invalid {field}: {value!r}")
    return date(parsed.year, parsed.month, parsed.day)


def _strict_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise DataReadinessError(f"transition boolean must be true or false, got {value!r}")


def _session_midnight(value: date) -> datetime:
    return datetime.combine(
        value,
        datetime.min.time(),
        tzinfo=ZoneInfo("America/New_York"),
    ).astimezone(UTC)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in frame.loc[:, list(_TRANSITION_COLUMNS)].to_dict(orient="records"):
        effective = pd.Timestamp(record["effective_at_utc"])
        record["effective_at_utc"] = effective.tz_convert("UTC").isoformat()
        record["identity_continuity"] = bool(record["identity_continuity"])
        record["membership_continuity"] = bool(record["membership_continuity"])
        records.append({str(key): value for key, value in record.items()})
    return records


def _records_sha256(frame: pd.DataFrame) -> str:
    return _json_sha256(_records(frame))


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _validate_window(start_date: date, cutoff_date: date) -> None:
    if start_date > cutoff_date:
        raise ValueError("start_date must not be after cutoff_date")


def _resolve_inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise DataReadinessError("S&P transition artifact escapes its authority directory")
    return candidate


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DataReadinessError(f"JSON artifact is not an object: {path}")
    return cast(dict[str, Any], value)


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
