"""Publish reviewed CIK issuer names, not membership, business segments or coverage."""
from __future__ import annotations

import json
import os
import re
import tomllib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import file_sha256, manifest_path_for, write_canonical_artifact
from market_predictor.catalysts.issuer_events.news_query_scope import load_news_query_scope
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, resolve_inside_authority, write_json_object
from market_predictor.universe.symbol_correction_policy import load_symbol_correction_policy

PUBLICATION_POLICY = "filing_date_next_day_new_york_publication_proxy"
_POLICY_INPUT = "issuer_identity_policy_sha256"
_LABEL_COLUMNS = (
    "security_id", "ticker", "company", "business_tag", "label_type", "match_terms",
    "tag_rank", "confidence", "relation_use", "effective_from_utc", "effective_to_utc", "available_at_utc",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Issuer(_Strict):
    security_id: str = Field(pattern=r"^cik:\d{10}$")
    ticker: str = Field(pattern=r"^[A-Z][A-Z.]{0,9}$")
    company: str = Field(min_length=1)
    aliases: tuple[str, ...] = Field(min_length=1)
    security_class: str = Field(min_length=1)
    identity_document_id: str
    publication_document_id: str
    publication_date_kind: Literal["as_filed_cover", "sec_filing_index"]
    record_locator: str = Field(min_length=20)


class _Policy(_Strict):
    schema_version: Literal["market_predictor.issuer_identity_publication.v1"]
    correction_policy: str
    correction_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_scope: str
    query_scope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    publication_inventory: str
    publication_archive: str
    publication_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    availability_policy: Literal["filing_date_next_day_new_york_publication_proxy"]
    production_ready: Literal[False]
    issuers: tuple[_Issuer, ...] = Field(min_length=2, max_length=2)


def _read(path: Path, maximum: int = 1024**2) -> bytes:
    if path.stat().st_size > maximum:
        raise DataReadinessError("issuer identity source exceeds size bound")
    return path.read_bytes()


def _document_bodies(root: Path, inventory_path: str, archive_path: str,
    report_sha256: str, wanted: set[str],
) -> dict[str, bytes]:
    from market_predictor.sources.official_documents import (
        load_official_document_inventory,
        verify_official_document_collection,
    )

    archive = inside(root, archive_path)
    if not archive.is_dir():
        raise DataReadinessError("issuer identity document archive is missing")
    inventory = load_official_document_inventory(resolve_inside_authority(root, inventory_path))
    report = verify_official_document_collection(archive, inventory)
    if report["status"] != "collected_unreviewed" or json_sha256(report) != report_sha256:
        raise DataReadinessError("issuer identity document report pin differs")
    bodies: dict[str, bytes] = {}
    for document in report["documents"]:
        if document["document_id"] not in wanted:
            continue
        attempts = [item for item in document["attempts"] if item["state"] == "archived_unreviewed"]
        if len(attempts) != 1:
            raise DataReadinessError("issuer identity needs one verified document body")
        path = resolve_inside_authority(archive, attempts[0]["receipt_path"])
        receipt = parse_strict_json_object(_read(path), label="official document receipt")
        body_name, response = receipt.get("body_path"), receipt.get("response")
        if not isinstance(body_name, str) or not isinstance(response, dict):
            raise DataReadinessError("issuer identity document receipt is malformed")
        body_path = resolve_inside_authority(archive, path.parent / body_name)
        if file_sha256(body_path) != response["sha256"]:
            raise DataReadinessError("issuer identity document body changed after replay")
        if response["content_encoding"] not in {None, "", "identity"}:
            raise DataReadinessError("issuer identity requires uncompressed retained HTML")
        bodies[document["document_id"]] = _read(body_path, 2 * 1024**2)
    return bodies


def _text(body: bytes) -> str:
    from bs4 import BeautifulSoup

    return " ".join(BeautifulSoup(body, "html.parser").get_text(" ", strip=True).split())


def _publication_proxy(body: bytes, kind: str) -> tuple[str, pd.Timestamp]:
    text = _text(body)
    if kind == "as_filed_cover":
        match = re.search(
            r"As filed with the Securities and Exchange Commission on ([A-Za-z]+ \d{1,2}, \d{4})",
            text[:2000],
        )
        if match is None:
            raise DataReadinessError("issuer source has no explicit as-filed date")
        filing_date = datetime.strptime(match[1], "%B %d, %Y").date()
    elif kind == "sec_filing_index":
        match = re.search(r"Filing Date (\d{4}-\d{2}-\d{2}) Accepted", text)
        if match is None:
            raise DataReadinessError("issuer source has no SEC index filing date")
        filing_date = datetime.strptime(match[1], "%Y-%m-%d").date()
    else:
        raise DataReadinessError("unsupported issuer publication-date evidence")
    # Date precision cannot establish an intraday availability time or first seen.
    available = pd.Timestamp(filing_date + timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC")
    return filing_date.isoformat(), available


def build_issuer_identity_inputs(*, root: Path, config: Path, expected_config_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Replay only small official sources and construct two identity-only rows."""
    from market_predictor.sources.official_documents import load_official_document_inventory

    root = root.resolve()
    config = resolve_inside_authority(root, config)
    if file_sha256(config) != expected_config_sha256:
        raise DataReadinessError("issuer identity policy requires its independent reviewed pin")
    policy = _Policy.model_validate(tomllib.loads(_read(config).decode("utf-8")))
    correction = load_symbol_correction_policy(root, Path(policy.correction_policy), policy.correction_policy_sha256)
    scope_path = resolve_inside_authority(root, policy.query_scope)
    if file_sha256(scope_path) != policy.query_scope_sha256:
        raise DataReadinessError("issuer identity query scope pin differs")
    scope, _ = load_news_query_scope(scope_path, root=root)
    keys = {(item.security_id, item.ticker) for item in policy.issuers}
    if len(keys) != 2 or keys != set(zip(scope["security_id"], scope["ticker"], strict=True)):
        raise DataReadinessError("issuer identity must match the complete corrected query scope")
    corrections = {item.security_id: item for item in correction.corrections}
    wanted = {value for item in policy.issuers for value in (item.identity_document_id, item.publication_document_id)}
    bodies = _document_bodies(root, correction.document_inventory, correction.document_archive,
        correction.document_report_sha256, wanted)
    additional = _document_bodies(root, policy.publication_inventory, policy.publication_archive,
        policy.publication_report_sha256, wanted)
    if set(bodies).intersection(additional):
        raise DataReadinessError("issuer identity has ambiguous publication document IDs")
    bodies.update(additional)
    if set(bodies) != wanted:
        raise DataReadinessError("issuer identity is missing reviewed official documents")
    rows: list[dict[str, Any]] = []
    for issuer in policy.issuers:
        reviewed = corrections.get(issuer.security_id)
        if (reviewed is None or issuer.identity_document_id not in reviewed.document_ids
                or issuer.ticker not in {reviewed.ticker, reviewed.provider_symbol}):
            raise DataReadinessError("issuer identity conflicts with reviewed symbol correction")
        body_text = _text(bodies[issuer.identity_document_id])
        for fact in (issuer.company, issuer.security_class, issuer.ticker, *issuer.aliases):
            if not re.search(r"(?<!\w)" + re.escape(fact) + r"(?!\w)", body_text, re.IGNORECASE):
                raise DataReadinessError("issuer name, alias, class or ticker is absent from its pinned source")
        publication = bodies[issuer.publication_document_id]
        if issuer.publication_date_kind == "sec_filing_index":
            index_text = _text(publication)
            if f"CIK : {issuer.security_id.removeprefix('cik:')}" not in index_text:
                raise DataReadinessError("issuer publication index belongs to a different CIK")
            # Pin the publication date to this body, not another filing by the issuer.
            inventory = load_official_document_inventory(resolve_inside_authority(root, correction.document_inventory))
            url = next(item.url for item in inventory.documents if item.document_id == issuer.identity_document_id)
            if url.rsplit("/", 1)[-1] not in index_text:
                raise DataReadinessError("issuer publication index does not identify the retained filing")
        filing_date, available = _publication_proxy(publication, issuer.publication_date_kind)
        interval = scope.loc[scope["security_id"].eq(issuer.security_id) & scope["ticker"].eq(issuer.ticker)].iloc[0]
        start, end = pd.Timestamp(interval["effective_from_utc"]), pd.Timestamp(interval["effective_to_utc"])
        if available > start:
            raise DataReadinessError("issuer pre-window filing proxy is later than the requested interval")
        rows.append({
            "security_id": issuer.security_id, "ticker": issuer.ticker, "company": issuer.company,
            "effective_from_utc": start, "effective_to_utc": end, "available_at_utc": available,
            "security_class": issuer.security_class, "official_aliases": json.dumps(issuer.aliases),
            "filing_date": filing_date, "availability_policy": PUBLICATION_POLICY,
            "historical_first_seen_proven": False, "source_coverage_admitted": False,
            "historical_disposition": "unknown_business_segments",
            "historical_label_count": 0, "historical_exposure_training_eligible": False,
            "record_locator": issuer.record_locator,
            "identity_document_id": issuer.identity_document_id,
            "publication_document_id": issuer.publication_document_id,
        })
    identities = pd.DataFrame(rows).sort_values("security_id").reset_index(drop=True)
    for column in ("effective_from_utc", "effective_to_utc", "available_at_utc"):
        identities[column] = identities[column].astype("datetime64[ns, UTC]")
    labels = pd.DataFrame({column: pd.Series(dtype="object") for column in _LABEL_COLUMNS})
    for column in ("effective_from_utc", "effective_to_utc", "available_at_utc"):
        labels[column] = pd.Series(dtype="datetime64[ns, UTC]")
    labels["tag_rank"] = pd.Series(dtype="int64")
    labels["confidence"] = pd.Series(dtype="float64")
    if file_sha256(config) != expected_config_sha256:
        raise DataReadinessError("issuer identity policy changed during replay")
    inputs = {_POLICY_INPUT: expected_config_sha256, "issuer_identity_policy_path": str(config),
        "issuer_identity_root": str(root), "availability_policy": PUBLICATION_POLICY,
        "business_label_policy": "unknown_no_inferred_segments"}
    return labels, identities, inputs


def verify_issuer_identity_inputs(labels: pd.DataFrame, identities: pd.DataFrame,
    label_manifest: dict[str, object], identity_manifest: dict[str, object],
) -> None:
    """Replay this publisher's evidence when its canonical inputs enter attribution."""
    label_inputs = label_manifest.get("inputs")
    identity_inputs = identity_manifest.get("inputs")
    inputs = [item for item in (label_inputs, identity_inputs) if isinstance(item, dict) and _POLICY_INPUT in item]
    if not inputs:
        return
    if len(inputs) != 2 or label_inputs != identity_inputs:
        raise DataReadinessError("issuer identity and empty label authority lineage differs")
    source = inputs[0]
    expected_labels, expected_identities, expected_inputs = build_issuer_identity_inputs(
        root=Path(str(source["issuer_identity_root"])), config=Path(str(source["issuer_identity_policy_path"])),
        expected_config_sha256=str(source[_POLICY_INPUT]),
    )
    if (source != expected_inputs or not labels.equals(expected_labels) or not identities.equals(expected_identities)
            or label_manifest.get("production_ready") is not False or identity_manifest.get("production_ready") is not False):
        raise DataReadinessError("issuer identity authority does not reproduce from pinned evidence")


def validate_issuer_event_scope(events: pd.DataFrame, identities: pd.DataFrame,
    identity_manifest: dict[str, object],
) -> None:
    """Do not let ticker-only matching bypass the new authority's reviewed dates."""
    inputs = identity_manifest.get("inputs")
    if not isinstance(inputs, dict) or _POLICY_INPUT not in inputs:
        return
    required = {"security_id", "ticker", "feature_available_at_utc"}
    if not required.issubset(events.columns):
        raise DataReadinessError("issuer event scope is missing source identity or clock")
    times = pd.to_datetime(events["feature_available_at_utc"], utc=True, errors="coerce")
    supported = pd.Series(False, index=events.index)
    for row in identities.to_dict(orient="records"):
        supported |= (
            events["security_id"].eq(row["security_id"]) & events["ticker"].eq(row["ticker"])
            & times.ge(row["effective_from_utc"]) & times.lt(row["effective_to_utc"])
            & times.ge(row["available_at_utc"])
        )
    if not supported.all():
        raise DataReadinessError("issuer event is outside the reviewed CIK, ticker or half-open publication scope")


def publish_issuer_identity_authority(*, root: Path, config: Path, expected_config_sha256: str,
    output_directory: Path,
) -> dict[str, str]:
    output = output_directory.resolve()
    if output.exists():
        raise DataReadinessError("issuer identity authority is immutable; choose a new directory")
    labels, identities, inputs = build_issuer_identity_inputs(
        root=root, config=config, expected_config_sha256=expected_config_sha256)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.{uuid4().hex}.pending")
    staging.mkdir()
    for name, frame, kind in (("business_labels", labels, "security_business_labels"),
        ("security_identities", identities, "security_business_label_coverage")):
        audit = CanonicalAuditReport(checks=(CanonicalAuditCheck(name="pinned_issuer_publication",
            status="pass", failures=0, rows_checked=len(frame),
            detail="Reviewed issuer-only publication proxy; business segments and source completeness remain unknown."),))
        path = staging / f"{name}.parquet"
        manifest = write_canonical_artifact(frame, path, artifact_type=kind,
            audit=audit, inputs=inputs, production_ready=False)
        manifest["artifact_path"] = str(output / path.name)
        final_manifest = staging / f"{name}.manifest.pending"
        write_json_object(final_manifest, manifest)
        final_manifest.replace(manifest_path_for(path))
    os.rename(staging, output)
    return {"business_labels": str(output / "business_labels.parquet"),
        "security_identities": str(output / "security_identities.parquet")}
