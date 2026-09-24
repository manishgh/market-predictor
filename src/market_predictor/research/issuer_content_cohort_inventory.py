"""Leased, resumable initial-fit cohort inventory of saved issuer news content."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from market_predictor.canonical.store import file_sha256, manifest_path_for
from market_predictor.catalysts.issuer_events.content_inventory import (
    inspect_saved_alpaca_content,
    verify_saved_alpaca_empty_chunk,
)
from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research.issuer_content_inventory import IMPLEMENTATION_PATHS as CHUNK_IMPLEMENTATION_PATHS
from market_predictor.research.issuer_content_inventory import _guard
from market_predictor.research.legacy_query_identity_proofs import load_identity_bridge, load_legacy_query_proofs, pin_file
from market_predictor.swing.contracts.research_cohort import load_swing_research_cohort
from market_predictor.swing.datasets.initial_fit_issuer_news import LAST_INITIAL_FIT_CUTOFF
from market_predictor.swing.datasets.initial_fit_issuer_news import _records as artifact_records
from market_predictor.swing.datasets.issuer_news_preparation import FIRST, _source
from market_predictor.swing.datasets.symbol_corrections import pinned_object
from market_predictor.universe.issuer_news_identity import map_news_coverage, map_news_relations
from market_predictor.universe.legacy_query_identity import (
    AVAILABILITY_BASIS,
    LEGACY_STATUS,
    PROOF_KINDS,
    map_legacy_query_coverage,
    map_legacy_query_relations,
)

CONFIG_SCHEMA = "market_predictor.issuer_content_cohort_inventory_config"
SCHEMA = "market_predictor.issuer_content_cohort_inventory"
ARCHIVES = ("early", "later", "corrected")
PART_CHUNKS = 200
ROW_GROUP_ROWS = 50_000
NEW_YORK = "America/New_York"
IMPLEMENTATION_PATHS = (
    *CHUNK_IMPLEMENTATION_PATHS, "research/issuer_content_cohort_inventory.py", "universe/issuer_news_identity.py",
    "universe/legacy_query_identity.py", "research/legacy_query_identity_proofs.py",
    "swing/contracts/research_cohort.py", "swing/datasets/issuer_news_publication.py",
    "swing/datasets/symbol_corrections.py", "catalysts/issuer_events/attribution.py",
    "catalysts/issuer_events/attribution_history.py",
)
_EVIDENCE = ("sidecar_pinned", "artifact_pinned_sidecar_observed", "derivation_unavailable_raw_pinned",
             "pages_verified_empty")
KNOWN_EMPTY_SCOPE = "provider_symbol_query_returned_no_items_not_proof_of_no_issuer_news"
EVIDENCE_LEVELS = {
    "sidecar_pinned": "artifact and sidecar pinned by the derivation's original source children",
    "artifact_pinned_sidecar_observed": "artifact pinned by the corrections manifest; sidecar observed and bound to it",
    "derivation_unavailable_raw_pinned": "artifact pinned by the raw collection manifest; sidecar observed and bound to it; "
                                         "old attribution/sentiment derivation missing",
    "pages_verified_empty": "saved pages bound to the pinned request admit no item; pages themselves are not pinned",
}
_ATTRIBUTED = ("bridged", "identity_equal", "proven_legacy_identity")
RESOLUTIONS = ("bridged", "bridged_non_cohort", "identity_equal", "proven_legacy_identity", "proven_legacy_non_cohort",
               "outside_bridge_span", "no_proven_identity")
# How an attributed query reaches its cohort security: the CIK bridge, the same ID, or a legacy proof kind.
BASES = ("cik_bridge", "identity_equal", *PROOF_KINDS)
_CONFIG_KEYS = {"schema", "monthly_news_config", "identity_manifest", "legacy_identity_proofs", "approved_population"}
_STATS = {"pages": "query_chunk_pages", "provider_rows": "query_chunk_provider_records",
          "accepted_rows": "query_chunk_admitted_stories"}
_CATEGORIES = ("provider_body_field", "provider_summary_field", "headline_only")
_SUMMARY_COLUMNS = ("archive", "chunk_id", "event_id", "query_security_id", "cohort_security_id", "query_identity_resolution",
                    "attribution_basis", "source_family", "provider_story_id", "inventory_status", "content_category",
                    "published_at_utc", "publication_year_new_york")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _anchored(root: Path, declared_root: str, value: str) -> str:
    """Re-express a saved absolute child path under this checkout; never a second root."""
    path = Path(value)
    _require(path.is_absolute() and path.is_relative_to(declared_root), "saved child escapes its declared repository")
    return inside(root, path.relative_to(declared_root)).relative_to(root).as_posix()


@dataclass(frozen=True, eq=False)
class _Identities:
    """Every pinned query-identity translation and the cohort it may attribute to."""

    bridge: pd.DataFrame
    proofs: pd.DataFrame
    cohort: frozenset[str]
    bridged: frozenset[str]


def _derived_units(root: Path, name: str, spec: dict[str, Any], source: dict[str, Any],
                   pins: dict[str, str]) -> list[dict[str, Any]]:
    directory = inside(root, spec["directory"])
    manifest = pinned_object(directory / "_manifest.json", spec["manifest_sha256"])
    pins[f"{directory.relative_to(root).as_posix()}/_manifest.json"] = spec["manifest_sha256"]
    resolution = manifest["source_path_resolution"]
    declared = resolution.get("repository_root")
    _require(resolution.get("policy") == "explicit_repository_base_authority_containment_v1" and isinstance(declared, str)
             and Path(declared).resolve() == root, "derived source must be read from its declared repository root")
    children = {_anchored(root, declared, key): pin for key, pin in
                pinned_object(directory / "_source_children.json", manifest["inventory"]["_source_children.json"]).items()}
    ledgers = [key for key in children if key.endswith("/_source_collections.parquet")]
    _require(len(ledgers) == 1, "derived source lacks one original coverage ledger")
    collection = ledgers[0].rsplit("/", 1)[0]
    raw_manifest = pinned_object(inside(root, f"{collection}/_manifest.json"), manifest["request"]["collection_manifest_sha256"])
    pins[f"{collection}/_manifest.json"] = manifest["request"]["collection_manifest_sha256"]
    raw_artifacts = artifact_records(raw_manifest)
    reasons = {gap["chunk_id"]: gap["reason"] for gap in manifest["unavailable_chunks"]}
    _require(len(reasons) == len(manifest["unavailable_chunks"]), "derived unavailable chunks are duplicated")
    units = []
    for row in source["ledger"].sort_values("chunk_id", kind="stable").to_dict("records"):
        chunk, events = str(row["chunk_id"]), f"{collection}/events/{row['chunk_id']}.parquet"
        unit = {"archive": name, "chunk_id": chunk, "collection": collection, "request_sha256": raw_manifest["request_sha256"],
                "derivation_status": row["status"], "derivation_reason": reasons.get(chunk), "ledger": row,
                "original_rows": int(row["original_row_count"]), "included_rows": int(row["row_count"]),
                "original_start": row["original_requested_start_utc"], "original_end": row["original_requested_end_utc"]}
        if row["status"] == "observed" and row["original_status"] == "observed":
            _require(chunk in raw_artifacts and children.get(events) == raw_artifacts[chunk]["sha256"]
                     and f"{events}.manifest.json" in children, "derived child and raw artifact pins differ")
            unit.update(status="observed", evidence="sidecar_pinned", artifact=SourcePin(path=events, sha256=children[events]),
                        sidecar=SourcePin(path=f"{events}.manifest.json", sha256=children[f"{events}.manifest.json"]))
        elif row["status"] == "unavailable_saved_evidence" and row["original_status"] == "observed":
            # Raw provider evidence exists; only the old attribution/sentiment derivation is missing.
            _require(chunk in reasons and chunk in raw_artifacts, "unavailable derived chunk lacks its reason or raw pin")
            unit.update(status="observed", evidence="derivation_unavailable_raw_pinned", included_rows=None,
                        artifact=SourcePin(path=events, sha256=raw_artifacts[chunk]["sha256"]))
        elif row["status"] == "observed_empty" and row["original_status"] == "observed_empty" and row["row_count"] == 0:
            unit.update(status="observed_empty", evidence="pages_verified_empty")
        else:
            raise DataReadinessError(f"derived coverage status is not inventoriable: {row['status']}")
        units.append(unit)
    return units


def _corrected_units(root: Path, spec: dict[str, Any], source: dict[str, Any], pins: dict[str, str]) -> list[dict[str, Any]]:
    for name in ("collection", "attribution", "sentiment"):
        pins[f"{source['roots'][name].relative_to(root).as_posix()}/_manifest.json"] = spec[name]["manifest_sha256"]
    pins[inside(root, spec["audit"]["path"]).relative_to(root).as_posix()] = spec["audit"]["sha256"]
    manifest = source["manifests"]["collection"]
    collection = source["roots"]["collection"].relative_to(root).as_posix()
    artifacts = source["records"]["collection"]
    units = []
    for row in source["ledger"].sort_values("chunk_id", kind="stable").to_dict("records"):
        chunk = str(row["chunk_id"])
        unit = {"archive": "corrected", "chunk_id": chunk, "collection": collection, "request_sha256": manifest["request_sha256"],
                "derivation_status": "not_derived", "derivation_reason": None, "ledger": row,
                "original_rows": int(row["row_count"]), "included_rows": None,
                "original_start": row["requested_start_utc"], "original_end": row["requested_end_utc"]}
        if row["status"] == "observed":
            events = f"{collection}/events/{chunk}.parquet"
            _require(chunk in artifacts, f"corrected observed chunk lacks its artifact pin: {chunk}")
            unit.update(status="observed", evidence="artifact_pinned_sidecar_observed",
                        artifact=SourcePin(path=events, sha256=artifacts[chunk]["sha256"]))
        elif row["status"] == "observed_empty" and row["row_count"] == 0:
            unit.update(status="observed_empty", evidence="pages_verified_empty")
        else:
            raise DataReadinessError(f"corrected coverage status is not inventoriable: {row['status']}")
        units.append(unit)
    return units


def _parity(unit: dict[str, Any], summary: dict[str, Any]) -> None:
    row, discarded = unit["ledger"], summary["query_chunk_discarded_records"]
    observed = {**{column: summary[key] for column, key in _STATS.items()},
        "duplicate_rows": summary["query_chunk_admitted_records"] - summary["query_chunk_admitted_stories"],
        "invalid_timestamp_rows": discarded["clock"] + discarded["title"],
        "outside_window_rows": discarded["window"], "symbol_mismatch_rows": discarded["symbol"]}
    _require(all(int(row[column]) == value for column, value in observed.items()), f"ledger producer statistics differ: {unit['chunk_id']}")
    _require(pd.Timestamp(summary["query_window_start_utc"]) == unit["original_start"]
             and pd.Timestamp(summary["query_window_end_exclusive_utc"]) == unit["original_end"],
             f"ledger query window differs: {unit['chunk_id']}")
    if unit["status"] == "observed":
        _require(summary["canonical_rows"] == unit["original_rows"], f"ledger canonical rows differ: {unit['chunk_id']}")
        _require(unit["included_rows"] is None or summary["included_rows"] == unit["included_rows"],
                 f"derived included rows differ: {unit['chunk_id']}")


def _resolution(status: pd.Series, target: pd.Series, original: pd.Series, identities: _Identities) -> pd.Series:
    """Query identity only, never issuer relevance: bridged, identity-equal, legacy-proven, then unresolved."""
    _require(bool(status.isin(("mapped", LEGACY_STATUS, "unmapped")).all()), "query identity status is unknown")
    mapped, legacy, unmapped = status.eq("mapped"), status.eq(LEGACY_STATUS), status.eq("unmapped")
    cohort, known = target.isin(identities.cohort), original.isin(identities.cohort)
    return pd.Series(pd.NA, index=status.index, dtype="string").mask(mapped & cohort, "bridged").mask(
        mapped & ~cohort, "bridged_non_cohort").mask(unmapped & known, "identity_equal").mask(
        legacy & cohort, "proven_legacy_identity").mask(legacy & ~cohort, "proven_legacy_non_cohort").mask(
        unmapped & ~known & original.isin(identities.bridged), "outside_bridge_span").fillna("no_proven_identity")


def _basis(resolution: pd.Series, kind: pd.Series) -> pd.Series:
    return pd.Series(pd.NA, index=resolution.index, dtype="string").mask(resolution.eq("bridged"), "cik_bridge").mask(
        resolution.eq("identity_equal"), "identity_equal").mask(resolution.eq("proven_legacy_identity"),
                                                                kind.astype("string"))


def _records(unit: dict[str, Any], rows: pd.DataFrame, identities: _Identities) -> pd.DataFrame:
    relations = pd.DataFrame({"relation_id": rows.event_id.astype(str), "target_security_id": rows.query_security_id.astype(str),
        "target_ticker": rows.query_ticker.astype(str), "event_feature_available_at_utc": rows.available_at_utc,
        "identity_available_at_utc": pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns, UTC]"),
        "feature_available_at_utc": rows.available_at_utc})
    mapped = map_legacy_query_relations(map_news_relations(relations, identities.bridge), identities.proofs)
    _require(mapped.identity_original_relation_id.tolist() == rows.event_id.astype(str).tolist(), "identity mapping reordered rows")
    resolution = _resolution(mapped.identity_translation_status, mapped.target_security_id,
                             mapped.identity_original_target_security_id, identities)
    complete = dict(zip(identities.proofs.proof_row_sha256, identities.proofs.evidence_complete_date, strict=True))
    result = rows.assign(archive=unit["archive"], evidence=unit["evidence"], derivation_status=unit["derivation_status"],
        query_identity_resolution=resolution.to_numpy(),
        identity_legacy_proof_evidence_complete_date=mapped.identity_legacy_proof_row_sha256.map(complete).to_numpy(),
        identity_bridge_row_sha256=mapped.identity_bridge_row_sha256.astype("string").to_numpy(),
        identity_legacy_proof_kind=mapped.identity_legacy_proof_kind.astype("string").to_numpy(),
        identity_legacy_proof_row_sha256=mapped.identity_legacy_proof_row_sha256.astype("string").to_numpy(),
        attribution_basis=_basis(resolution, mapped.identity_legacy_proof_kind).to_numpy(),
        cohort_security_id=mapped.target_security_id.astype("string").where(resolution.isin(_ATTRIBUTED).to_numpy()).to_numpy(),
        publication_year_new_york=rows.published_at_utc.dt.tz_convert(NEW_YORK).dt.year.astype("int16"))
    for column in ("archive", "evidence", "derivation_status", "query_identity_resolution", "identity_bridge_row_sha256",
                   "identity_legacy_proof_kind", "identity_legacy_proof_row_sha256",
                   "identity_legacy_proof_evidence_complete_date", "attribution_basis", "cohort_security_id"):
        result[column] = result[column].astype("string")
    return result


def _unit_summary(unit: dict[str, Any], summary: dict[str, Any], records: pd.DataFrame | None) -> dict[str, Any]:
    keys = ("query_chunk_pages", "query_chunk_provider_records", "query_chunk_discarded_records",
            "query_chunk_admitted_stories", "canonical_rows", "included_rows", "version_after_cutoff_rows",
            "outside_publication_window_rows", "query_security_id", "query_ticker", "query_provider_symbol", "source_files")
    resolutions = records.query_identity_resolution.value_counts().to_dict() if records is not None else {}
    return {"archive": unit["archive"], "chunk_id": unit["chunk_id"], "status": unit["status"], "evidence": unit["evidence"],
            "derivation_status": unit["derivation_status"], "derivation_reason": unit["derivation_reason"],
            **{key: summary[key] for key in keys if key in summary},
            "known_empty_scope": KNOWN_EMPTY_SCOPE if unit["status"] == "observed_empty" else None,
            "record_resolutions": {str(key): int(value) for key, value in sorted(resolutions.items())}}


def _inspect(root: Path, unit: dict[str, Any], identities: _Identities) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    row = unit["ledger"]
    if unit["status"] == "observed_empty":
        summary = verify_saved_alpaca_empty_chunk(root=root, collection=Path(unit["collection"]), chunk_id=unit["chunk_id"],
            collection_request_sha256=unit["request_sha256"], security_id=str(row["security_id"]), ticker=str(row["ticker"]),
            memory_check=_guard)
        _parity(unit, summary)
        return None, _unit_summary(unit, summary, None)
    artifact = unit["artifact"]
    sidecar = unit.get("sidecar")
    if sidecar is None:
        # These sidecars carry no external pin; bind them to the pinned artifact and request.
        path = manifest_path_for(inside(root, artifact.path))
        declared = pinned_object(path)
        _require(declared.get("artifact_sha256") == artifact.sha256
                 and declared.get("inputs", {}).get("collection_request_sha256") == unit["request_sha256"],
                 f"unpinned sidecar does not bind its pinned artifact and request: {unit['chunk_id']}")
        sidecar = SourcePin(path=path.relative_to(root).as_posix(), sha256=file_sha256(path))
    rows, summary = inspect_saved_alpaca_content(root=root, event_artifact=artifact, event_manifest=sidecar,
        security_id=str(row["security_id"]), ticker=str(row["ticker"]), start_utc=FIRST, cutoff_utc=LAST_INITIAL_FIT_CUTOFF,
        memory_check=_guard)
    _parity(unit, summary)
    records = _records(unit, rows, identities)
    return records, _unit_summary(unit, summary, records)


# Same pattern as research/swing_return_training.py; importing it would load estimators, and moving it
# would change bytes pinned by closed training evidence.
def _atomic_json(output: Path, name: str, payload: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(dir=output, prefix=".checkpoint-") as temporary:
        staged = Path(temporary) / name
        write_json_object(staged, payload)
        os.replace(staged, output / name)


def _coverage(units: list[dict[str, Any]], summaries: dict[str, dict[str, Any]], identities: _Identities) -> pd.DataFrame:
    _require(set(summaries) == {f"{unit['archive']}/{unit['chunk_id']}" for unit in units}
             and len(summaries) == len(units), "every ledger unit requires exactly one verified summary")
    frames = []
    for archive in ARCHIVES:
        selected = [unit for unit in units if unit["archive"] == archive]
        ledger = pd.DataFrame([unit["ledger"] for unit in selected])
        requested = ledger.assign(requested_start_utc=ledger.requested_start_utc.clip(lower=FIRST),
                                  requested_end_utc=ledger.requested_end_utc.clip(upper=LAST_INITIAL_FIT_CUTOFF))
        _require(bool((requested.requested_start_utc < requested.requested_end_utc).all()),
                 "every ledger unit must cover part of the initial-fit window")
        segments = map_legacy_query_coverage(map_news_coverage(
            requested[["chunk_id", "security_id", "ticker", "requested_start_utc", "requested_end_utc"]], identities.bridge),
            identities.proofs)
        _require(set(segments.identity_original_chunk_id) == set(requested.chunk_id), "coverage segments lost a ledger unit")
        segments["archive"] = archive
        frames.append(segments)
    result = pd.concat(frames, ignore_index=True)
    key = result.archive + "/" + result.identity_original_chunk_id
    for column in ("status", "evidence", "derivation_status", "query_provider_symbol"):
        result[column] = key.map(lambda value, name=column: summaries[value][name]).astype("string")
    result["query_identity_resolution"] = _resolution(result.identity_translation_status, result.security_id,
                                                      result.identity_original_security_id, identities).to_numpy()
    result["attribution_basis"] = _basis(result.query_identity_resolution, result.identity_legacy_proof_kind).to_numpy()
    result["cohort_security_id"] = result.security_id.astype("string").where(result.query_identity_resolution.isin(_ATTRIBUTED))
    attributed = result.loc[result.cohort_security_id.notna()].sort_values(["cohort_security_id", "requested_start_utc"])
    overlaps = attributed.groupby("cohort_security_id").requested_start_utc.shift(-1) < attributed.requested_end_utc
    _require(not bool(overlaps.fillna(False).any()), "attributed coverage overlaps for one cohort security")
    return result.sort_values(["archive", "identity_original_chunk_id", "requested_start_utc"], kind="stable").reset_index(drop=True)


def _years() -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    spans = []
    for year in range(FIRST.tz_convert(NEW_YORK).year, LAST_INITIAL_FIT_CUTOFF.tz_convert(NEW_YORK).year + 1):
        start = pd.Timestamp(year=year, month=1, day=1, tz=NEW_YORK).tz_convert("UTC")
        end = pd.Timestamp(year=year + 1, month=1, day=1, tz=NEW_YORK).tz_convert("UTC")
        spans.append((year, max(start, FIRST), min(end, LAST_INITIAL_FIT_CUTOFF)))
    return spans


def _security_years(records: pd.DataFrame, coverage: pd.DataFrame, cohort: tuple[str, ...]) -> pd.DataFrame:
    attributed = records.loc[records.cohort_security_id.notna()].assign(
        _rank=lambda frame: frame.inventory_status.map({"included": 0, "version_after_cutoff": 1}),
        _archive=lambda frame: frame.archive.map({name: index for index, name in enumerate(ARCHIVES)}))
    # One representative per story: included before after-cutoff, then archive, chunk and event order.
    stories = attributed.sort_values(["cohort_security_id", "source_family", "provider_story_id", "_rank", "_archive",
                                      "chunk_id", "event_id"], kind="stable").drop_duplicates(
        ["cohort_security_id", "source_family", "provider_story_id"])
    key = [stories.cohort_security_id, stories.publication_year_new_york]
    included = stories.inventory_status.eq("included")
    counts = pd.DataFrame({"query_returned_stories": stories.groupby(key).size(),
        "included_stories": included.groupby(key).sum(),
        "version_after_cutoff_stories": stories.inventory_status.eq("version_after_cutoff").groupby(key).sum(),
        **{f"included_{category}": (included & stories.content_category.eq(category)).groupby(key).sum()
           for category in _CATEGORIES},
        **{f"included_stories_by_{basis}": (included & stories.attribution_basis.eq(basis)).groupby(key).sum()
           for basis in BASES}})
    covered = coverage.loc[coverage.cohort_security_id.notna()]
    spans = []
    for year, start, end in _years():
        lower = covered.requested_start_utc.clip(lower=start)
        upper = covered.requested_end_utc.clip(upper=end)
        spans.append(covered.assign(year_new_york=year,
                                    days=((upper - lower).dt.total_seconds() / 86_400).clip(lower=0).to_numpy()))
    segments = pd.concat(spans, ignore_index=True)
    active = segments.loc[segments.days > 0]
    _require(bool(active.attribution_basis.isin(BASES).all()), "attributed coverage lacks its attribution basis")
    by_evidence = active.groupby(["cohort_security_id", "year_new_york", "evidence"]).days.sum()
    by_basis = active.groupby(["cohort_security_id", "year_new_york", "attribution_basis"]).days.sum()
    tickers = active.groupby(["cohort_security_id", "year_new_york"]).ticker.agg(lambda values: ",".join(sorted(set(values))))
    rows = []
    for security in cohort:
        for year, start, end in _years():
            window = (end - start).total_seconds() / 86_400
            days = {name: float(by_evidence.get((security, year, name), 0.0)) for name in _EVIDENCE}
            count = counts.loc[(security, year)] if (security, year) in counts.index else None
            rows.append({"cohort_security_id": security, "source_family": "alpaca", "year_new_york": year,
                "query_tickers": tickers.get((security, year), ""),
                **{column: int(count[column]) if count is not None else 0 for column in counts.columns},
                "window_days": window, **{f"covered_{name}_days": value for name, value in days.items()},
                **{f"covered_by_{basis}_days": float(by_basis.get((security, year, basis), 0.0)) for basis in BASES},
                "unknown_no_proven_query_days": max(0.0, window - sum(days.values())),
                "attribution_status": "not_established", "known_empty_scope": KNOWN_EMPTY_SCOPE})
    return pd.DataFrame(rows)


def _totals(records: pd.DataFrame, coverage: pd.DataFrame, summaries: dict[str, dict[str, Any]],
            cohort: tuple[str, ...]) -> dict[str, Any]:
    units = pd.DataFrame(summaries.values())
    segments = coverage.assign(prefix=coverage.identity_original_security_id.str.split(":").str[0],
        days=(coverage.requested_end_utc - coverage.requested_start_utc).dt.total_seconds() / 86_400)
    unattributed = segments.loc[segments.cohort_security_id.isna()]
    rows = records.assign(prefix=records.query_security_id.str.split(":").str[0])
    reached = set(coverage.cohort_security_id.dropna())
    mapped = records.loc[records.query_identity_resolution.isin(("bridged", "proven_legacy_identity"))
                         & records.inventory_status.eq("included"),
                         ["archive", "chunk_id", "event_id", "cohort_security_id", "published_at_utc"]]
    spans = coverage.loc[coverage.cohort_security_id.notna(), ["archive", "identity_original_chunk_id", "cohort_security_id",
                                                               "requested_start_utc", "requested_end_utc"]]
    joined = mapped.merge(spans.rename(columns={"identity_original_chunk_id": "chunk_id"}),
                          on=["archive", "chunk_id", "cohort_security_id"], how="left")
    inside_segment = joined.requested_start_utc.le(joined.published_at_utc) & joined.published_at_utc.lt(joined.requested_end_utc)
    exceptions = len(mapped) - int(inside_segment.groupby([joined.archive, joined.chunk_id, joined.event_id]).any().sum())
    return {"units": {f"{a}/{s}/{e}": int(n) for (a, s, e), n in units.groupby(["archive", "status", "evidence"]).size().items()},
        "derivation_unavailable": {str(k): int(v) for k, v in
            units.loc[units.derivation_status.eq("unavailable_saved_evidence")].groupby("derivation_reason").size().items()},
        "records": {f"{a}/{i}/{s}": int(n) for (a, i, s), n in
            rows.groupby(["archive", "query_identity_resolution", "inventory_status"]).size().items()},
        "unattributed_coverage_segments": {f"{a}/{i}/{p}": int(n) for (a, i, p), n in
            unattributed.groupby(["archive", "query_identity_resolution", "prefix"]).size().items()},
        "unattributed_coverage_days": {f"{a}/{i}/{p}": float(n) for (a, i, p), n in
            unattributed.groupby(["archive", "query_identity_resolution", "prefix"]).days.sum().items()},
        "unattributed_query_ids": {str(k): int(v) for k, v in
            unattributed.groupby("query_identity_resolution").identity_original_security_id.nunique().items()},
        "unattributed_records": {f"{a}/{i}/{p}": int(n) for (a, i, p), n in
            rows.loc[rows.cohort_security_id.isna()].groupby(["archive", "query_identity_resolution", "prefix"]).size().items()},
        "cohort_securities": len(cohort), "cohort_securities_with_proven_query": len(reached & set(cohort)),
        "cohort_securities_without_proven_query": len(set(cohort) - reached),
        "translated_included_records_outside_their_coverage_segment": exceptions,
        "attributed_records_by_basis": {f"{a}/{b}/{s}": int(n) for (a, b, s), n in
            rows.groupby(["archive", "attribution_basis", "inventory_status"]).size().items()},
        "attributed_coverage_days_by_basis": {str(b): float(n) for b, n in
            segments.loc[segments.cohort_security_id.notna()].groupby("attribution_basis").days.sum().items()},
        "legacy_proven_query_ids": {str(k): int(v) for k, v in
            segments.loc[segments.identity_translation_status.eq(LEGACY_STATUS)].groupby(
                "query_identity_resolution").identity_original_security_id.nunique().items()}}


def publish_cohort_content_inventory(*, root: Path, config: Path, config_sha256: str, output: Path,
                                     resume_checkpoint_sha256: str | None = None) -> dict[str, Any]:
    """Inventory every pinned initial-fit query unit; never content qualification or admission."""
    root = root.resolve()
    config, output = inside(root, config), inside(root, output)
    _require(output.parent == root / "data/research", "cohort inventory output must be a direct child of data/research")
    runtime = heavy_job_runtime_dir()
    if not runtime.is_absolute():
        runtime = root / runtime
    with heavy_job_lease("inspect-issuer-content-cohort", runtime_dir=runtime, config_path=config):
        _guard()
        pins: dict[str, str] = {}
        settings = pinned_object(pin_file(root, {"path": config.relative_to(root).as_posix(), "sha256": config_sha256}, pins),
                                 config_sha256)
        _require(set(settings) == _CONFIG_KEYS and settings["schema"] == CONFIG_SCHEMA, "cohort inventory configuration differs")
        monthly = pinned_object(pin_file(root, settings["monthly_news_config"], pins), settings["monthly_news_config"]["sha256"])
        specs = monthly.get("sources")
        if not isinstance(specs, dict) or set(specs) != set(ARCHIVES):
            raise DataReadinessError("monthly news sources differ")
        bridge, _ = load_identity_bridge(root, settings["identity_manifest"], pins)
        proofs, proof_manifest = load_legacy_query_proofs(root, settings["legacy_identity_proofs"], pins)
        identity_path = inside(root, settings["identity_manifest"]["path"]).relative_to(root).as_posix()
        _require(proof_manifest["source_files"].get(identity_path) == settings["identity_manifest"]["sha256"],
                 "legacy identity proofs were built against another CIK bridge")
        _require(proof_manifest.get("availability_basis") == AVAILABILITY_BASIS, "legacy identity proof clock differs")
        population = pin_file(root, settings["approved_population"], pins)
        cohort_ids = load_swing_research_cohort(population, source_root=root).retained_security_ids
        identities = _Identities(bridge, proofs, frozenset(cohort_ids), frozenset(bridge.source_security_id))
        units: list[dict[str, Any]] = []
        for name in ARCHIVES:
            _guard()
            source = _source(root, specs[name])
            units += (_derived_units(root, name, specs[name], source, pins) if name != "corrected"
                      else _corrected_units(root, specs[name], source, pins))
        package = Path(__file__).resolve().parents[1]
        implementation = {f"market_predictor/{name}": file_sha256(package / name) for name in IMPLEMENTATION_PATHS}
        request = {"schema": f"{SCHEMA}_request", "config": {"path": config.relative_to(root).as_posix(), "sha256": config_sha256},
            "window": {"start_utc": FIRST.isoformat(), "cutoff_utc": LAST_INITIAL_FIT_CUTOFF.isoformat()},
            "source_files": dict(sorted(pins.items())), "implementation_files": implementation, "part_chunks": PART_CHUNKS,
            "units_sha256": json_sha256([[unit["archive"], unit["chunk_id"], unit["status"], unit["evidence"]] for unit in units]),
            "training_eligible": False, "serving_eligible": False, "promotion_eligible": False}
        _require(not (output / "_manifest.json").exists(), "completed cohort inventory is immutable")
        output.mkdir(exist_ok=True)
        request_path, checkpoint_path = output / "_request.json", output / "_checkpoint.json"
        parts: dict[str, Any] = {}
        if request_path.exists():
            _require(resume_checkpoint_sha256 is not None, "resuming requires an independently pinned checkpoint SHA256")
            _require(pinned_object(request_path) == request, "cohort inventory request differs; use a new output directory")
            checkpoint = pinned_object(checkpoint_path, resume_checkpoint_sha256)
            _require(checkpoint.get("request_sha256") == file_sha256(request_path), "cohort inventory checkpoint request differs")
            parts = checkpoint["parts"]
        else:
            _require(resume_checkpoint_sha256 is None and not any(output.iterdir()),
                     "cohort inventory output has files without a bound request")
            write_json_object(request_path, request)
            _atomic_json(output, "_checkpoint.json", {"request_sha256": file_sha256(request_path), "parts": {}})
        request_sha256 = file_sha256(request_path)
        (output / "parts").mkdir(exist_ok=True)
        summaries: dict[str, dict[str, Any]] = {}
        for name in ARCHIVES:
            selected = [unit for unit in units if unit["archive"] == name]
            for index in range(0, len(selected), PART_CHUNKS):
                part = f"{name}-{index // PART_CHUNKS:04d}"
                batch = selected[index:index + PART_CHUNKS]
                records_path, units_path = output / "parts" / f"{part}.parquet", output / "parts" / f"{part}.units.json"
                if part in parts:
                    _require(file_sha256(records_path) == parts[part]["records_sha256"]
                             and file_sha256(units_path) == parts[part]["units_sha256"], f"checkpointed part changed: {part}")
                    saved = pinned_object(units_path)["units"]
                    _require([entry["chunk_id"] for entry in saved] == [unit["chunk_id"] for unit in batch],
                             f"checkpointed part covers other chunks: {part}")
                else:
                    frames, saved = [], []
                    for unit in batch:
                        records, summary = _inspect(root, unit, identities)
                        saved.append(summary)
                        if records is not None:
                            frames.append(records)
                    _guard()
                    for path in (records_path, units_path):
                        path.unlink(missing_ok=True)
                    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                    frame.to_parquet(records_path, index=False, row_group_size=ROW_GROUP_ROWS)
                    write_json_object(units_path, {"request_sha256": request_sha256, "units": saved})
                    parts[part] = {"records_sha256": file_sha256(records_path), "units_sha256": file_sha256(units_path)}
                    _atomic_json(output, "_checkpoint.json", {"request_sha256": request_sha256, "parts": parts})
                summaries.update({f"{name}/{entry['chunk_id']}": entry for entry in saved})
        _require(set(parts) == {path.name.removesuffix(".units.json") for path in (output / "parts").glob("*.units.json")},
                 "cohort inventory parts differ from its checkpoint")
        _guard()
        paths = [output / "parts" / f"{part}.parquet" for part in sorted(parts)]
        records = pd.concat([pd.read_parquet(path, columns=list(_SUMMARY_COLUMNS)) for path in paths
                             if pq.ParquetFile(path).metadata.num_rows], ignore_index=True)  # type: ignore[no-untyped-call]
        coverage = _coverage(units, summaries, identities)
        years = _security_years(records, coverage, cohort_ids)
        for frame, name in ((coverage, "coverage.parquet"), (years, "security_years.parquet")):
            path = output / name
            path.unlink(missing_ok=True)
            frame.to_parquet(path, index=False)
        result = {"schema": SCHEMA, "status": "complete_inventory_only", "request_sha256": request_sha256,
            "window": request["window"], "parts": dict(sorted(parts.items())), "records_rows": len(records),
            "coverage_sha256": file_sha256(output / "coverage.parquet"),
            "security_years_sha256": file_sha256(output / "security_years.parquet"),
            "totals": _totals(records, coverage, summaries, cohort_ids),
            "count_scope": "stories_returned_by_issuer_queries_not_issuer_relevance", "known_empty_scope": KNOWN_EMPTY_SCOPE,
            "evidence_levels": EVIDENCE_LEVELS,
            "legacy_proof_availability_basis": AVAILABILITY_BASIS,
            "legacy_proof_scope": "proven_legacy_identity attribution starts at the membership effective start, a "
                                  "retrospective clock; each record carries its proof's evidence-complete date; "
                                  "research evidence only",
            "zero_story_scope": "zero stories inside observed windows are also query-level evidence only",
            "attribution_status": "not_established", "content_qualification": "not_established",
            "training_eligible": False, "serving_eligible": False, "promotion_eligible": False}
        final = output / "_manifest.json"
        _atomic_json(output, "_manifest.json", json.loads(json.dumps(result, default=str)))
        return {**result, "manifest_sha256": file_sha256(final)}
