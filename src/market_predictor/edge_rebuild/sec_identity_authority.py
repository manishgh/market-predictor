"""Hash-bound SEC issuer identity for the point-in-time S&P universe.

This authority is intentionally offline.  It consumes an explicitly supplied
SEC ``company_tickers.json`` snapshot and never downloads or guesses identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, Protocol, cast
from urllib.parse import urlsplit
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.sp500_memberships import (
    require_sp500_membership_authority,
)
from market_predictor.edge_rebuild.sp500_transitions import (
    require_sp500_transition_authority,
)
from market_predictor.v3.contracts import normalized_ticker
from market_predictor.v3.errors import DataReadinessError

SEC_IDENTITY_REQUEST_SCHEMA: Final = "edge_rebuild.sec_identity_request.v2"
SEC_IDENTITY_MANIFEST_SCHEMA: Final = "edge_rebuild.sec_identity_manifest.v2"
SEC_IDENTITY_AUTHORITY_SCHEMA: Final = "edge_rebuild.sec_identity_authority.v2"
SEC_IDENTITY_RELATION_SCHEMA: Final = "edge_rebuild.sec_identity_relations.v2"
SEC_IDENTITY_COVERAGE_SCHEMA: Final = "edge_rebuild.sec_identity_coverage.v2"
RELATION_FILE: Final = "sec_identity_relations.parquet"
COVERAGE_FILE: Final = "sec_identity_coverage.csv"
DEFAULT_RELATION_START: Final = date(2019, 7, 9)
DEFAULT_RELATION_END: Final = date(2026, 7, 8)
DEFAULT_MEMBERSHIP_START: Final = date(2018, 5, 29)
MAXIMUM_WHOLE_SECURITY_EXCLUSION_FRACTION: Final = 0.05
_ET = ZoneInfo("America/New_York")
_RELATION_COLUMNS: Final = (
    "security_id",
    "ticker",
    "sec_cik",
    "effective_from_utc",
    "effective_to_utc",
    "available_at_utc",
    "proof_ticker",
    "proof_company_name",
    "identity_policy",
    "schema_version",
)
_COVERAGE_COLUMNS: Final = (
    "security_id",
    "latest_ticker",
    "first_effective_from_utc",
    "last_effective_to_utc",
    "interval_count",
    "ticker_count",
    "tickers",
    "status",
    "reason",
    "sec_cik",
    "proof_ticker",
)
_OVERRIDE_COLUMNS: Final = (
    "security_id",
    "ticker",
    "issuer_name",
    "sec_cik",
    "effective_from_utc",
    "effective_to_utc",
    "evidence_url",
    "evidence_document",
    "evidence_accession",
    "evidence_raw_sha256",
    "reviewer_status",
    "reason",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ACCESSION = re.compile(r"^\d{10}-\d{2}-\d{6}$")


class _ScalarWithItem(Protocol):
    def item(self) -> object: ...


@dataclass(frozen=True, slots=True)
class SecIdentityConfig:
    relation_start_date: date = DEFAULT_RELATION_START
    relation_end_date: date = DEFAULT_RELATION_END
    membership_start_date: date = DEFAULT_MEMBERSHIP_START
    maximum_whole_security_exclusion_fraction: float = MAXIMUM_WHOLE_SECURITY_EXCLUSION_FRACTION


@dataclass(frozen=True, slots=True)
class SecIdentityAuthority:
    directory: Path
    relations: pd.DataFrame
    coverage: pd.DataFrame
    manifest: Mapping[str, object]
    authority: Mapping[str, object]


def load_sec_identity_config(path: Path) -> SecIdentityConfig:
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(f"SEC identity config is unreadable: {path}") from exc
    section = payload.get("sec_identity")
    if not isinstance(section, dict):
        raise DataReadinessError("SEC identity config requires [sec_identity]")
    try:
        config = SecIdentityConfig(
            relation_start_date=date.fromisoformat(str(section["relation_start_date"])),
            relation_end_date=date.fromisoformat(str(section["relation_end_date"])),
            membership_start_date=date.fromisoformat(str(section["membership_start_date"])),
            maximum_whole_security_exclusion_fraction=float(section["maximum_whole_security_exclusion_fraction"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DataReadinessError("SEC identity config contains invalid values") from exc
    return validate_sec_identity_config(config)


def validate_sec_identity_config(config: SecIdentityConfig) -> SecIdentityConfig:
    if config.relation_start_date != DEFAULT_RELATION_START:
        raise DataReadinessError("SEC identity relation must start on 2019-07-09")
    if config.relation_end_date != DEFAULT_RELATION_END:
        raise DataReadinessError("SEC identity relation must end on 2026-07-08")
    if config.membership_start_date > config.relation_start_date:
        raise DataReadinessError("membership authority must start before the SEC relation")
    if config.maximum_whole_security_exclusion_fraction != 0.05:
        raise DataReadinessError("SEC identity whole-security exclusion ceiling is frozen at 5%")
    return config


def load_sec_company_ticker_mapping(path: Path) -> pd.DataFrame:
    """Read and strictly validate a local raw SEC company-ticker snapshot."""

    if not path.is_file():
        raise DataReadinessError(f"local SEC company ticker mapping is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"SEC company ticker mapping is unreadable: {path}") from exc
    if not isinstance(payload, dict) or not payload:
        raise DataReadinessError("SEC company ticker mapping must be a non-empty object")
    rows: list[dict[str, str]] = []
    for key, value in payload.items():
        if not isinstance(value, dict):
            raise DataReadinessError(f"SEC company ticker row {key!r} is malformed")
        try:
            ticker = _sec_ticker(str(value.get("ticker", "")))
            cik = _cik(value.get("cik_str"))
        except ValueError as exc:
            raise DataReadinessError(f"SEC company ticker row {key!r} is invalid") from exc
        rows.append(
            {
                "mapping_row_id": str(key),
                "ticker": ticker,
                "sec_cik": cik,
                "company_name": str(value.get("title", "")).strip(),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["ticker", "sec_cik", "mapping_row_id"], kind="stable")
    conflicts = frame.groupby("ticker", sort=False)["sec_cik"].nunique()
    if bool(conflicts.gt(1).any()):
        symbols = ", ".join(conflicts[conflicts.gt(1)].index.astype(str))
        raise DataReadinessError(f"SEC mapping assigns conflicting CIKs to: {symbols}")
    return frame.drop_duplicates(["ticker", "sec_cik"], keep="first").reset_index(drop=True)


def load_reviewed_sec_identity_overrides(path: Path) -> pd.DataFrame:
    """Load reviewed issuer identities and verify their archived SEC evidence."""

    if not path.is_file():
        raise DataReadinessError(f"reviewed SEC identity override ledger is missing: {path}")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, ValueError) as exc:
        raise DataReadinessError(f"reviewed SEC identity override ledger is unreadable: {path}") from exc
    missing = sorted(set(_OVERRIDE_COLUMNS).difference(frame.columns))
    if missing:
        raise DataReadinessError("reviewed SEC identity override ledger missing columns: " + ", ".join(missing))
    output = frame.loc[:, list(_OVERRIDE_COLUMNS)].copy()
    for column in _OVERRIDE_COLUMNS:
        output[column] = output[column].astype(str).str.strip()
    if output.empty:
        return output
    if bool(output[["security_id", "issuer_name", "reason"]].eq("").any(axis=None)):
        raise DataReadinessError("reviewed SEC identity override has blank identity or review evidence")
    output["ticker"] = output["ticker"].map(_sec_ticker)
    output["sec_cik"] = output["sec_cik"].map(_cik)
    output["effective_from_utc"] = pd.to_datetime(output["effective_from_utc"], utc=True, errors="raise")
    output["effective_to_utc"] = pd.to_datetime(output["effective_to_utc"], utc=True, errors="raise")
    if bool(output["effective_from_utc"].ge(output["effective_to_utc"]).any()):
        raise DataReadinessError("reviewed SEC identity override has an invalid effective interval")
    if bool(output["reviewer_status"].ne("approved").any()):
        raise DataReadinessError("reviewed SEC identity overrides must have reviewer_status=approved")
    if bool(output["evidence_url"].map(lambda value: not _official_sec_url(value)).any()):
        raise DataReadinessError("reviewed SEC identity override evidence must use an official SEC URL")
    if bool(output["evidence_accession"].map(lambda value: _ACCESSION.fullmatch(value) is None).any()):
        raise DataReadinessError("reviewed SEC identity override has an invalid filing accession")
    if bool(output["evidence_raw_sha256"].map(lambda value: _SHA256.fullmatch(value.lower()) is None).any()):
        raise DataReadinessError("reviewed SEC identity override has an invalid evidence SHA-256")
    if bool(output.duplicated("security_id", keep=False).any()):
        raise DataReadinessError("reviewed SEC identity override assigns a security more than once")
    if bool(output.duplicated("ticker", keep=False).any()):
        raise DataReadinessError("reviewed SEC identity override ticker is ambiguous or reused")
    for row in output.itertuples(index=False):
        evidence = (path.parent / str(row.evidence_document)).resolve()
        if not evidence.is_file():
            raise DataReadinessError(f"reviewed SEC identity evidence is missing for {row.security_id}")
        if file_sha256(evidence).lower() != str(row.evidence_raw_sha256).lower():
            raise DataReadinessError(f"reviewed SEC identity evidence hash mismatch for {row.security_id}")
        evidence_url = urlsplit(str(row.evidence_url))
        expected_archive_prefix = (
            f"/Archives/edgar/data/{int(str(row.sec_cik))}/{str(row.evidence_accession).replace('-', '')}/"
        )
        if not evidence_url.path.startswith(expected_archive_prefix):
            raise DataReadinessError(f"reviewed SEC identity evidence URL mismatch for {row.security_id}")
        if not _filing_names_trading_symbol(evidence, str(row.ticker)):
            raise DataReadinessError(f"reviewed SEC filing does not name trading symbol for {row.security_id}")
    output["evidence_raw_sha256"] = output["evidence_raw_sha256"].str.lower()
    return output.sort_values("security_id", kind="stable").reset_index(drop=True)


def build_sec_identity_relations(
    memberships: pd.DataFrame,
    transitions: pd.DataFrame,
    sec_mapping: pd.DataFrame,
    reviewed_overrides: pd.DataFrame | None = None,
    *,
    config: SecIdentityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Map the latest ticker, then propagate only within one stable security ID."""

    config = validate_sec_identity_config(config)
    membership = _normalize_memberships(memberships, config=config)
    mapping = _normalize_mapping_frame(sec_mapping)
    transition = _normalize_transitions(transitions)
    overrides = _normalize_override_frame(reviewed_overrides)
    mapping_by_ticker = mapping.set_index("ticker", drop=False).to_dict("index")
    override_by_security = overrides.set_index("security_id", drop=False).to_dict("index")
    ticker_security_counts = membership.groupby("ticker")["security_id"].nunique()
    unknown_override_ids = sorted(set(override_by_security).difference(membership["security_id"].astype(str)))
    if unknown_override_ids:
        raise DataReadinessError("reviewed SEC identity override references unknown security_id: " + ", ".join(unknown_override_ids))
    relation_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for security_id, group in membership.groupby("security_id", sort=True):
        ordered = group.sort_values("effective_from_utc", kind="stable").reset_index(drop=True)
        latest_ticker = str(ordered.iloc[-1]["ticker"])
        tickers = list(dict.fromkeys(ordered["ticker"].astype(str)))
        override = override_by_security.get(str(security_id))
        proof = override if override is not None else mapping_by_ticker.get(latest_ticker)
        reason: str | None = None
        if override is not None:
            _validate_override_for_security(
                override,
                ordered,
                transition,
                str(security_id),
                latest_ticker,
                int(ticker_security_counts.get(latest_ticker, 0)),
            )
        elif proof is None:
            reason = "latest_ticker_absent_from_sec_mapping"
        elif (embedded_cik := _embedded_cik(str(security_id))) is not None and embedded_cik != str(proof["sec_cik"]):
            reason = "stable_security_id_cik_conflicts_with_sec_mapping"
        elif int(ticker_security_counts.get(latest_ticker, 0)) > 1:
            reason = "ticker_reused_by_different_security_id"
        elif not _ticker_chain_is_proven(ordered, transition, str(security_id)):
            reason = "ticker_transition_identity_not_proven"
        if reason is None and proof is not None:
            cik = str(proof["sec_cik"])
            for row in ordered.itertuples(index=False):
                relation_rows.append(
                    {
                        "security_id": str(security_id),
                        "ticker": str(row.ticker),
                        "sec_cik": cik,
                        "effective_from_utc": row.effective_from_utc,
                        "effective_to_utc": row.effective_to_utc,
                        "available_at_utc": row.effective_from_utc,
                        "proof_ticker": latest_ticker,
                        "proof_company_name": str(proof.get("issuer_name", proof.get("company_name", ""))),
                        "identity_policy": (
                            "reviewed_official_sec_filing_override_v1"
                            if override is not None
                            else "latest_sec_ticker_propagated_within_stable_security_id_v1"
                        ),
                        "schema_version": SEC_IDENTITY_RELATION_SCHEMA,
                    }
                )
        coverage_rows.append(
            {
                "security_id": str(security_id),
                "latest_ticker": latest_ticker,
                "first_effective_from_utc": ordered["effective_from_utc"].min(),
                "last_effective_to_utc": ordered["effective_to_utc"].max(),
                "interval_count": len(ordered),
                "ticker_count": len(tickers),
                "tickers": "|".join(tickers),
                "status": "resolved" if reason is None else "excluded",
                "reason": "" if reason is None else reason,
                "sec_cik": "" if proof is None or reason is not None else str(proof["sec_cik"]),
                "proof_ticker": "" if proof is None else latest_ticker,
            }
        )
    relations = pd.DataFrame(relation_rows, columns=list(_RELATION_COLUMNS))
    if not relations.empty:
        relations = relations.sort_values(["security_id", "effective_from_utc", "ticker"], kind="stable").reset_index(drop=True)
    coverage = pd.DataFrame(coverage_rows, columns=list(_COVERAGE_COLUMNS)).sort_values("security_id", kind="stable").reset_index(drop=True)
    total = len(coverage)
    excluded = int(coverage["status"].eq("excluded").sum())
    share = excluded / total if total else 1.0
    summary: dict[str, object] = {
        "schema": SEC_IDENTITY_COVERAGE_SCHEMA,
        "security_count": total,
        "resolved_security_count": total - excluded,
        "excluded_security_count": excluded,
        "excluded_security_fraction": share,
        "maximum_whole_security_exclusion_fraction": config.maximum_whole_security_exclusion_fraction,
        "coverage_passed": total > 0 and share <= config.maximum_whole_security_exclusion_fraction,
        "relation_intervals": len(relations),
        "issuer_cik_count": int(relations["sec_cik"].nunique()) if not relations.empty else 0,
        "dual_class_cik_count": _dual_class_cik_count(relations),
        "unresolved_security_ids": coverage.loc[coverage["status"].eq("excluded"), "security_id"].astype(str).tolist(),
    }
    return relations, coverage, summary


def publish_sec_identity_authority(
    *,
    sec_mapping_path: Path,
    reviewed_overrides_path: Path,
    membership_directory: Path,
    archive_directory: Path,
    event_directory: Path,
    transition_directory: Path,
    reviewed_transitions_path: Path,
    anchor_path: Path,
    output_directory: Path,
    config: SecIdentityConfig,
) -> SecIdentityAuthority:
    """Publish a new immutable authority after complete offline parent replay."""

    config = validate_sec_identity_config(config)
    output = output_directory.resolve()
    if output.exists():
        raise DataReadinessError(f"SEC identity authority is immutable: {output}")
    memberships, transitions, mapping, overrides, parent = _verified_inputs(
        sec_mapping_path=sec_mapping_path,
        reviewed_overrides_path=reviewed_overrides_path,
        membership_directory=membership_directory,
        archive_directory=archive_directory,
        event_directory=event_directory,
        transition_directory=transition_directory,
        reviewed_transitions_path=reviewed_transitions_path,
        anchor_path=anchor_path,
        config=config,
    )
    relations, coverage, summary = build_sec_identity_relations(
        memberships,
        transitions,
        mapping,
        overrides,
        config=config,
    )
    if not bool(summary["coverage_passed"]):
        unresolved = int(cast(int, summary["excluded_security_count"]))
        total = int(cast(int, summary["security_count"]))
        raise DataReadinessError(
            f"SEC identity excludes {unresolved} of {total} whole securities ({unresolved / total:.2%}), above the frozen 5.00% ceiling"
        )
    request_payload = _request_payload(parent, config)
    request_sha256 = _json_sha256(request_payload)
    staging = output.with_name(f".{output.name}.{uuid4().hex}.staging")
    staging.mkdir(parents=True)
    try:
        relation_path = staging / RELATION_FILE
        coverage_path = staging / COVERAGE_FILE
        relations.to_parquet(relation_path, index=False, compression="zstd")
        coverage.to_csv(coverage_path, index=False, lineterminator="\n")
        request = {**request_payload, "request_sha256": request_sha256}
        _write_json(staging / "_request.json", request)
        manifest = {
            "schema": SEC_IDENTITY_MANIFEST_SCHEMA,
            "status": "complete",
            "request_sha256": request_sha256,
            "parent_lineage": parent,
            "relation_artifact": _artifact_record(relation_path),
            "coverage_artifact": _artifact_record(coverage_path),
            "relation_semantic_sha256": _frame_sha256(relations, _RELATION_COLUMNS),
            "coverage_semantic_sha256": _frame_sha256(coverage, _COVERAGE_COLUMNS),
            "coverage": summary,
        }
        _write_json(staging / "_manifest.json", manifest)
        authority = {
            "schema": SEC_IDENTITY_AUTHORITY_SCHEMA,
            "state": "identity_complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(staging / "_manifest.json"),
            "request_sha256": request_sha256,
            "relation_semantic_sha256": manifest["relation_semantic_sha256"],
            "coverage_semantic_sha256": manifest["coverage_semantic_sha256"],
            "security_count": summary["security_count"],
            "excluded_security_count": summary["excluded_security_count"],
        }
        _write_json(staging / "_authority.json", authority)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return require_sec_identity_authority(
        output,
        sec_mapping_path=sec_mapping_path,
        reviewed_overrides_path=reviewed_overrides_path,
        membership_directory=membership_directory,
        archive_directory=archive_directory,
        event_directory=event_directory,
        transition_directory=transition_directory,
        reviewed_transitions_path=reviewed_transitions_path,
        anchor_path=anchor_path,
        config=config,
    )


def require_sec_identity_authority(
    authority_directory: Path,
    *,
    sec_mapping_path: Path,
    reviewed_overrides_path: Path,
    membership_directory: Path,
    archive_directory: Path,
    event_directory: Path,
    transition_directory: Path,
    reviewed_transitions_path: Path,
    anchor_path: Path,
    config: SecIdentityConfig,
) -> SecIdentityAuthority:
    """Replay all parents and the effective-dated relation offline."""

    config = validate_sec_identity_config(config)
    memberships, transitions, mapping, overrides, parent = _verified_inputs(
        sec_mapping_path=sec_mapping_path,
        reviewed_overrides_path=reviewed_overrides_path,
        membership_directory=membership_directory,
        archive_directory=archive_directory,
        event_directory=event_directory,
        transition_directory=transition_directory,
        reviewed_transitions_path=reviewed_transitions_path,
        anchor_path=anchor_path,
        config=config,
    )
    expected_request = _request_payload(parent, config)
    request_sha256 = _json_sha256(expected_request)
    request = _load_json(authority_directory / "_request.json")
    if request != {**expected_request, "request_sha256": request_sha256}:
        raise DataReadinessError("SEC identity request or parent lineage is invalid")
    authority = _load_json(authority_directory / "_authority.json")
    manifest_path = authority_directory / str(authority.get("artifact", ""))
    if (
        authority.get("schema") != SEC_IDENTITY_AUTHORITY_SCHEMA
        or authority.get("state") != "identity_complete"
        or not manifest_path.is_file()
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
    ):
        raise DataReadinessError("SEC identity authority is invalid")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema") != SEC_IDENTITY_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or manifest.get("parent_lineage") != parent
        or authority.get("request_sha256") != request_sha256
    ):
        raise DataReadinessError("SEC identity manifest lineage is invalid")
    relation_path = _verified_artifact(authority_directory, manifest.get("relation_artifact"))
    coverage_path = _verified_artifact(authority_directory, manifest.get("coverage_artifact"))
    actual_relations = _read_relations(relation_path)
    actual_coverage = _read_coverage(coverage_path)
    expected_relations, expected_coverage, expected_summary = build_sec_identity_relations(
        memberships,
        transitions,
        mapping,
        overrides,
        config=config,
    )
    relation_hash = _frame_sha256(actual_relations, _RELATION_COLUMNS)
    coverage_hash = _frame_sha256(actual_coverage, _COVERAGE_COLUMNS)
    if (
        _records(actual_relations, _RELATION_COLUMNS) != _records(expected_relations, _RELATION_COLUMNS)
        or _records(actual_coverage, _COVERAGE_COLUMNS) != _records(expected_coverage, _COVERAGE_COLUMNS)
        or manifest.get("coverage") != expected_summary
        or manifest.get("relation_semantic_sha256") != relation_hash
        or manifest.get("coverage_semantic_sha256") != coverage_hash
        or authority.get("relation_semantic_sha256") != relation_hash
        or authority.get("coverage_semantic_sha256") != coverage_hash
    ):
        raise DataReadinessError("SEC identity relation does not replay from bound inputs")
    return SecIdentityAuthority(
        directory=authority_directory,
        relations=actual_relations,
        coverage=actual_coverage,
        manifest=manifest,
        authority=authority,
    )


def _verified_inputs(
    *,
    sec_mapping_path: Path,
    reviewed_overrides_path: Path,
    membership_directory: Path,
    archive_directory: Path,
    event_directory: Path,
    transition_directory: Path,
    reviewed_transitions_path: Path,
    anchor_path: Path,
    config: SecIdentityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    memberships = require_sp500_membership_authority(
        membership_directory,
        archive_directory=archive_directory,
        event_directory=event_directory,
        transition_directory=transition_directory,
        reviewed_transitions_path=reviewed_transitions_path,
        anchor_path=anchor_path,
        start_date=config.membership_start_date,
        cutoff_date=config.relation_end_date,
    )
    transitions = require_sp500_transition_authority(
        transition_directory,
        archive_directory=archive_directory,
        event_directory=event_directory,
        reviewed_transitions_path=reviewed_transitions_path,
        start_date=config.membership_start_date,
        cutoff_date=config.relation_end_date,
    )
    mapping = load_sec_company_ticker_mapping(sec_mapping_path)
    overrides = load_reviewed_sec_identity_overrides(reviewed_overrides_path)
    membership_authority = _load_json(membership_directory / "_authority.json")
    transition_authority = _load_json(transition_directory / "_authority.json")
    parent = {
        "sec_company_tickers": {
            "sha256": file_sha256(sec_mapping_path),
            "rows": len(mapping),
        },
        "reviewed_sec_identity_overrides": {
            "sha256": file_sha256(reviewed_overrides_path),
            "rows": len(overrides),
            "evidence_set_sha256": _override_evidence_set_sha256(overrides),
        },
        "membership_authority": {
            "authority_sha256": file_sha256(membership_directory / "_authority.json"),
            "universe_sha256": membership_authority.get("universe_sha256"),
        },
        "transition_authority": {
            "authority_sha256": file_sha256(transition_directory / "_authority.json"),
            "transition_set_sha256": transition_authority.get("transition_set_sha256"),
        },
        "reviewed_transitions": {
            "sha256": file_sha256(reviewed_transitions_path),
        },
    }
    return memberships, transitions, mapping, overrides, parent


def _normalize_memberships(frame: pd.DataFrame, *, config: SecIdentityConfig) -> pd.DataFrame:
    required = {
        "security_id",
        "ticker",
        "effective_from_utc",
        "effective_to_utc",
        "available_at_utc",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataReadinessError("membership authority missing columns: " + ", ".join(missing))
    output = frame.loc[:, list(required)].copy()
    output["security_id"] = output["security_id"].astype(str).str.strip()
    output["ticker"] = output["ticker"].map(_sec_ticker)
    output["effective_from_utc"] = pd.to_datetime(output["effective_from_utc"], utc=True, errors="raise")
    output["effective_to_utc"] = pd.to_datetime(output["effective_to_utc"], utc=True, errors="coerce")
    output["available_at_utc"] = pd.to_datetime(output["available_at_utc"], utc=True, errors="raise")
    start = pd.Timestamp(config.relation_start_date, tz=_ET).tz_convert("UTC")
    end = (pd.Timestamp(config.relation_end_date, tz=_ET) + pd.Timedelta(days=1)).tz_convert("UTC")
    output = output.loc[
        output["effective_from_utc"].lt(end) & (output["effective_to_utc"].isna() | output["effective_to_utc"].gt(start))
    ].copy()
    output["effective_from_utc"] = output["effective_from_utc"].where(output["effective_from_utc"].ge(start), start)
    output["effective_to_utc"] = output["effective_to_utc"].where(
        output["effective_to_utc"].notna() & output["effective_to_utc"].le(end), end
    )
    if output.empty or bool(output["security_id"].eq("").any()):
        raise DataReadinessError("membership authority has no valid in-window securities")
    if bool(output.duplicated(["security_id", "effective_from_utc"]).any()):
        raise DataReadinessError("membership security intervals are ambiguous")
    for _, group in output.sort_values(["security_id", "effective_from_utc"], kind="stable").groupby("security_id", sort=False):
        starts = group["effective_from_utc"].iloc[1:].reset_index(drop=True)
        ends = group["effective_to_utc"].iloc[:-1].reset_index(drop=True)
        if bool(starts.lt(ends).any()):
            raise DataReadinessError("membership security intervals overlap")
    return output.sort_values(["security_id", "effective_from_utc", "ticker"], kind="stable").reset_index(drop=True)


def _normalize_mapping_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "sec_cik", "company_name"}
    if not required.issubset(frame.columns):
        raise DataReadinessError("SEC mapping frame is incomplete")
    output = frame.loc[:, list(required)].copy()
    output["ticker"] = output["ticker"].map(_sec_ticker)
    output["sec_cik"] = output["sec_cik"].map(_cik)
    output["company_name"] = output["company_name"].fillna("").astype(str).str.strip()
    conflicts = output.groupby("ticker")["sec_cik"].nunique()
    if bool(conflicts.gt(1).any()):
        raise DataReadinessError("SEC mapping contains conflicting ticker identities")
    return output.drop_duplicates("ticker", keep="first").sort_values("ticker", kind="stable").reset_index(drop=True)


def _normalize_override_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(columns=list(_OVERRIDE_COLUMNS))
    missing = sorted(set(_OVERRIDE_COLUMNS).difference(frame.columns))
    if missing:
        raise DataReadinessError("reviewed SEC identity override frame missing columns: " + ", ".join(missing))
    output = frame.loc[:, list(_OVERRIDE_COLUMNS)].copy()
    if output.empty:
        return output
    for column in _OVERRIDE_COLUMNS:
        output[column] = output[column].astype(str).str.strip()
    output["ticker"] = output["ticker"].map(_sec_ticker)
    output["sec_cik"] = output["sec_cik"].map(_cik)
    output["effective_from_utc"] = pd.to_datetime(output["effective_from_utc"], utc=True, errors="raise")
    output["effective_to_utc"] = pd.to_datetime(output["effective_to_utc"], utc=True, errors="raise")
    if bool(output.duplicated("security_id", keep=False).any()):
        raise DataReadinessError("reviewed SEC identity override assigns a security more than once")
    if bool(output.duplicated("ticker", keep=False).any()):
        raise DataReadinessError("reviewed SEC identity override ticker is ambiguous or reused")
    return output.sort_values("security_id", kind="stable").reset_index(drop=True)


def _validate_override_for_security(
    override: Mapping[str, object],
    intervals: pd.DataFrame,
    transitions: pd.DataFrame,
    security_id: str,
    latest_ticker: str,
    ticker_security_count: int,
) -> None:
    if str(override["security_id"]) != security_id:
        raise DataReadinessError("reviewed SEC identity override security_id mismatch")
    if str(override["ticker"]) != latest_ticker:
        raise DataReadinessError(f"reviewed SEC identity override ticker mismatch for {security_id}")
    if ticker_security_count != 1:
        raise DataReadinessError(f"reviewed SEC identity override ticker is reused for {security_id}")
    if pd.Timestamp(override["effective_from_utc"]) != intervals["effective_from_utc"].min() or pd.Timestamp(
        override["effective_to_utc"]
    ) != intervals["effective_to_utc"].max():
        raise DataReadinessError(f"reviewed SEC identity override interval mismatch for {security_id}")
    embedded_cik = _embedded_cik(security_id)
    if embedded_cik is not None and embedded_cik != str(override["sec_cik"]):
        raise DataReadinessError(f"reviewed SEC identity override conflicts with embedded CIK for {security_id}")
    if not _ticker_chain_is_proven(intervals, transitions, security_id):
        raise DataReadinessError(f"reviewed SEC identity override has an unproven ticker transition for {security_id}")


def _override_evidence_set_sha256(frame: pd.DataFrame) -> str:
    return _frame_sha256(frame, _OVERRIDE_COLUMNS)


def _official_sec_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme == "https" and parsed.hostname in {"www.sec.gov", "data.sec.gov"}


def _filing_names_trading_symbol(path: Path, ticker: str) -> bool:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise DataReadinessError(f"reviewed SEC identity evidence is unreadable: {path}") from exc
    plain = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw))
    symbol = re.escape(ticker)
    return re.search(
        rf"(?i)(trading\s+symbol|symbol).{{0,500}}\b{symbol}\b|\b{symbol}\b.{{0,500}}(trading\s+symbol|symbol)",
        plain,
    ) is not None


def _normalize_transitions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "effective_at_utc",
        "old_ticker",
        "new_ticker",
        "identity_continuity",
        "old_security_id",
        "new_security_id",
    }
    if not required.issubset(frame.columns):
        raise DataReadinessError("transition authority is incomplete")
    output = frame.loc[:, list(required)].copy()
    output["effective_at_utc"] = pd.to_datetime(output["effective_at_utc"], utc=True, errors="raise")
    output["old_ticker"] = output["old_ticker"].map(_sec_ticker)
    output["new_ticker"] = output["new_ticker"].map(_sec_ticker)
    output["identity_continuity"] = output["identity_continuity"].astype(bool)
    for column in ("old_security_id", "new_security_id"):
        output[column] = output[column].fillna("").astype(str).str.strip()
    return output


def _ticker_chain_is_proven(intervals: pd.DataFrame, transitions: pd.DataFrame, security_id: str) -> bool:
    if len(intervals) < 2:
        return True
    for previous, current in zip(
        intervals.iloc[:-1].itertuples(index=False),
        intervals.iloc[1:].itertuples(index=False),
        strict=True,
    ):
        if previous.ticker == current.ticker:
            continue
        boundary = pd.Timestamp(current.effective_from_utc)
        candidates = transitions.loc[
            transitions["effective_at_utc"].eq(boundary)
            & transitions["identity_continuity"]
            & (
                (transitions["old_ticker"].eq(previous.ticker) & transitions["new_ticker"].eq(current.ticker))
                | (transitions["old_ticker"].eq(current.ticker) & transitions["new_ticker"].eq(previous.ticker))
            )
        ]
        if candidates.empty:
            return False
        explicit = candidates.loc[candidates["old_security_id"].ne("") | candidates["new_security_id"].ne("")]
        if not explicit.empty and not bool(
            (explicit["old_security_id"].eq(security_id) & explicit["new_security_id"].eq(security_id)).any()
        ):
            return False
    return True


def _request_payload(parent: Mapping[str, object], config: SecIdentityConfig) -> dict[str, object]:
    return {
        "schema": SEC_IDENTITY_REQUEST_SCHEMA,
        "relation_start_date": config.relation_start_date.isoformat(),
        "relation_end_date": config.relation_end_date.isoformat(),
        "membership_start_date": config.membership_start_date.isoformat(),
        "maximum_whole_security_exclusion_fraction": config.maximum_whole_security_exclusion_fraction,
        "identity_policy": "official_sec_mapping_plus_reviewed_filing_overrides_v2",
        "ticker_reuse_policy": "never_cross_security_id",
        "parent_lineage": parent,
    }


def _read_relations(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise DataReadinessError("SEC identity relation artifact is unreadable") from exc
    missing = sorted(set(_RELATION_COLUMNS).difference(frame.columns))
    if missing:
        raise DataReadinessError("SEC identity relation artifact is incomplete")
    for column in ("effective_from_utc", "effective_to_utc", "available_at_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    return (
        frame.loc[:, list(_RELATION_COLUMNS)]
        .sort_values(["security_id", "effective_from_utc", "ticker"], kind="stable")
        .reset_index(drop=True)
    )


def _read_coverage(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    except (OSError, ValueError) as exc:
        raise DataReadinessError("SEC identity coverage artifact is unreadable") from exc
    if not set(_COVERAGE_COLUMNS).issubset(frame.columns):
        raise DataReadinessError("SEC identity coverage artifact is incomplete")
    for column in ("first_effective_from_utc", "last_effective_to_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    for column in ("interval_count", "ticker_count"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    return frame.loc[:, list(_COVERAGE_COLUMNS)].sort_values("security_id", kind="stable").reset_index(drop=True)


def _verified_artifact(directory: Path, record: object) -> Path:
    if not isinstance(record, dict):
        raise DataReadinessError("SEC identity artifact inventory is invalid")
    path = (directory / str(record.get("path", ""))).resolve()
    root = directory.resolve()
    if root not in path.parents or not path.is_file():
        raise DataReadinessError("SEC identity artifact path is invalid")
    if record.get("sha256") != file_sha256(path) or int(record.get("bytes", -1)) != path.stat().st_size:
        raise DataReadinessError("SEC identity artifact hash is invalid")
    return path


def _artifact_record(path: Path) -> dict[str, object]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _dual_class_cik_count(relations: pd.DataFrame) -> int:
    if relations.empty:
        return 0
    pairs = relations[["sec_cik", "security_id"]].drop_duplicates()
    return int(pairs.groupby("sec_cik")["security_id"].nunique().gt(1).sum())


def _records(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[dict[str, object]]:
    return [{column: _json_value(row[column]) for column in columns} for row in frame.loc[:, list(columns)].to_dict(orient="records")]


def _frame_sha256(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    return _json_sha256(_records(frame, columns))


def _json_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return cast(_ScalarWithItem, value).item()
    return value


def _cik(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith("cik:"):
        text = text[4:]
    if not text.isdigit() or len(text) > 10:
        raise ValueError("invalid SEC CIK")
    return text.zfill(10)


def _embedded_cik(security_id: str) -> str | None:
    parts = security_id.strip().lower().split(":")
    if len(parts) < 2 or parts[0] != "cik" or not parts[1].isdigit():
        return None
    try:
        return _cik(parts[1])
    except ValueError:
        return None


def _sec_ticker(value: str) -> str:
    return normalized_ticker(value.strip().upper().replace("-", "."))


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"SEC identity metadata is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"SEC identity metadata is not an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
