"""Leased per-issuer SEC acceptance-clock evidence: sampled EDGAR detail pages, then the published conventions.

The page collection fetches each issuer's first and last archive filing in every form group.
The publication compares those pages with the saved raw acceptance values, adds in-data
counts from filings dated before the initial-fit cutoff's New York date, and decides one
convention per issuer (`catalysts.sec_filings.acceptance_clock`).
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.sec_filings import document_collection as documents
from market_predictor.catalysts.sec_filings.acceptance_clock import (
    CONVENTIONS,
    UNKNOWN,
    clock_sample,
    decide,
    form_group,
    in_data_counts,
    page_convention,
)
from market_predictor.catalysts.sec_filings.collection import (
    load_sec_filing_collection,
    replay_sec_filing_collection,
    saved_filings,
)
from market_predictor.config import Settings
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research.legacy_query_identity_proofs import pin_file
from market_predictor.research.sec_page_collection import RUNNER_PATHS, memory_guard, open_collection, run_collection
from market_predictor.sources.http import HttpClient
from market_predictor.sources.sec import SecSource
from market_predictor.swing.datasets.initial_fit_issuer_news import LAST_INITIAL_FIT_CUTOFF
from market_predictor.swing.datasets.symbol_corrections import pinned_object

SCHEMA = "market_predictor.sec_acceptance_clock"
PAGE_SCHEMA = "market_predictor.sec_acceptance_clock_pages"
PHASES = ("header",)
# Filings dated before this New York day are available by the initial-fit cutoff under either reading.
EVIDENCE_BEFORE = LAST_INITIAL_FIT_CUTOFF.tz_convert("America/New_York").date()
IMPLEMENTATION_PATHS = (
    "research/sec_acceptance_clock.py", "catalysts/sec_filings/acceptance_clock.py", "catalysts/sec_filings/collection.py",
)
_FILING_COLUMNS = ["sec_cik", "accession_number", "sec_form", "filing_date", "acceptance_raw"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _guard() -> None:
    memory_guard("SEC acceptance clock")


def archive_source(client: HttpClient) -> SecSource:
    """The unchanged SEC client over saved responses; the configured identity is validated, never sent."""
    return SecSource(Settings(), client=client)


def _archive(root: Path, collection: dict[str, str], pins: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """Every archive event with its raw acceptance, replayed from the saved responses, and every issuer."""
    authority = pin_file(root, collection, pins)
    loaded = load_sec_filing_collection(authority.parent)
    _require(authority.name == "_authority.json" and dict(loaded.authority) == pinned_object(authority),
             "SEC collection authority pin differs")
    rows: list[dict[str, str]] = []
    for position, issuer in enumerate(replay_sec_filing_collection(loaded, archive_source)):
        if position % 25 == 0:
            _guard()
        saved = saved_filings(issuer.history)
        for accession in issuer.events.accession_number.astype(str):
            record, row = saved[accession]
            rows.append({"sec_cik": issuer.cik, "accession_number": accession, "sec_form": record.form,
                         "filing_date": record.filing_date, "acceptance_raw": str(row.get("acceptanceDateTime") or "")})
    return pd.DataFrame(rows, columns=_FILING_COLUMNS), sorted(loaded.source_collections.sec_cik.astype(str))


def _page_units(sample: pd.DataFrame) -> pd.DataFrame:
    """One detail page per sampled accession, read from its first filer's folder."""
    first = sample.sort_values(["accession_number", "sec_cik"], kind="stable").drop_duplicates("accession_number")
    return pd.DataFrame({"sec_cik": first.sec_cik.astype(str), "accession_number": first.accession_number.astype(str),
                         "url": [documents.index_url(cik, accession) for cik, accession in
                                 zip(first.sec_cik, first.accession_number, strict=True)]}).reset_index(drop=True)


def _collection_pin(root: Path, collection: dict[str, str]) -> dict[str, str]:
    return {"path": inside(root, collection["path"]).relative_to(root).as_posix(), "sha256": collection["sha256"]}


def collect_sec_clock_pages(*, root: Path, collection: dict[str, str], output: Path,
                            resume_checkpoint_sha256: str | None = None, settings: Settings | None = None) -> dict[str, Any]:
    """Collect the detail page of each issuer's first and last archive filing in every form group."""

    def plan(pins: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
        filings, _ = _archive(root.resolve(), collection, pins)
        return _page_units(clock_sample(filings)), {
            "sec_collection": _collection_pin(root.resolve(), collection),
            "selection": "first and last archive filing of every issuer in every form group; one detail page per accession",
        }

    return run_collection(root=root, output=output, job="collect-sec-clock-pages", schema=PAGE_SCHEMA, plan=plan,
                          phases=PHASES, implementation_paths=IMPLEMENTATION_PATHS,
                          resume_checkpoint_sha256=resume_checkpoint_sha256, settings=settings)


def _pages(store: documents.Store, units: pd.DataFrame) -> pd.DataFrame:
    """Each page unit's final state and, for an archived page, EDGAR's own acceptance. A page for another
    accession fails; any other unreadable page leaves its groups without evidence."""
    final, receipts = documents.outcomes(store, units, PHASES)
    store.verify(receipts)
    rejected = receipts.loc[receipts.state.eq("rejected")]
    foreign = rejected.loc[rejected.reason.fillna("").str.contains("another accession")]
    _require(foreign.empty, f"EDGAR served detail pages of other accessions: {sorted(foreign.accession_number)[:5]}")
    reasons = dict(zip(rejected.accession_number.astype(str), rejected.reason.astype(str), strict=True))
    archived = receipts.loc[receipts.state.eq("archived")]
    accepted: dict[str, pd.Timestamp] = {}
    for shard, rows in archived.groupby("shard", sort=True):
        with zipfile.ZipFile(store.output / "shards" / f"{shard}.zip") as archive:
            for row in rows.itertuples(index=False):
                accession = str(row.accession_number)
                header = documents.parse_filing_header(archive.read(str(row.member)), accession=accession)
                accepted[accession] = header.accepted_at_utc
    return pd.DataFrame({"accession_number": final.accession_number.astype(str), "page_state": final.state.astype(str),
                         "page_reason": [reasons.get(str(accession)) for accession in final.accession_number],
                         "page_accepted_utc": [accepted.get(str(accession), pd.NaT) for accession in final.accession_number]})


def publish_sec_acceptance_clock(*, root: Path, collection: dict[str, str], pages: dict[str, str], output: Path
                                 ) -> dict[str, Any]:
    """Decide and publish every archive issuer's acceptance-clock convention; never a guess."""
    root = root.resolve()
    output = inside(root, output)
    _require(output.parent == root / "data/research", "SEC acceptance clock must be a new direct child of data/research")
    runtime = heavy_job_runtime_dir()
    if not runtime.is_absolute():
        runtime = root / runtime
    with heavy_job_lease("publish-sec-acceptance-clock", runtime_dir=runtime):
        _guard()
        if output.exists():
            raise FileExistsError(f"immutable SEC acceptance clock already exists: {output.relative_to(root).as_posix()}")
        pins: dict[str, str] = {}
        filings, issuers = _archive(root, collection, pins)
        sample = clock_sample(filings)
        units = _page_units(sample)
        store, manifest = open_collection(root, pages, PAGE_SCHEMA)
        pin_file(root, pages, pins)
        request = pinned_object(store.output / "_request.json")
        _require(request.get("request_sha256") == manifest["request_sha256"]
                 and request.get("sec_collection") == _collection_pin(root, collection)
                 and request.get("work_list_sha256") == json_sha256(units.to_dict("records")),
                 "SEC clock pages were collected for another archive or sample")
        _guard()
        compared = sample.merge(_pages(store, units), on="accession_number", how="left", validate="many_to_one")
        compared["page_convention"] = [
            page_convention(raw, accepted) if state == "archived" else None
            for raw, state, accepted in zip(compared.acceptance_raw, compared.page_state, compared.page_accepted_utc,
                                            strict=True)]
        counts = in_data_counts(filings, EVIDENCE_BEFORE)
        groups = filings.assign(form_group=filings.sec_form.map(form_group)).groupby("sec_cik").form_group.agg(set).to_dict()
        pages_by_issuer = dict(tuple(compared.groupby("sec_cik")))
        counts_by_issuer = dict(tuple(counts.groupby("sec_cik")))
        rows = []
        for cik in issuers:
            if cik not in groups:
                convention, reason = UNKNOWN, "no archive filings"
            else:
                convention, reason = decide(cik, groups[cik], pages_by_issuer[cik], counts_by_issuer.get(cik, counts.iloc[:0]))
            own = compared.loc[compared.sec_cik.eq(cik) & compared.page_state.eq("archived")]
            rows.append({"sec_cik": cik, "convention": convention, "reason": reason,
                         "form_groups": ",".join(sorted(groups.get(cik, ()))),
                         "archive_filings": int(filings.sec_cik.eq(cik).sum()), "pages_compared": len(own),
                         "pages_after_initial_fit_cutoff": int(own.page_accepted_utc.gt(LAST_INITIAL_FIT_CUTOFF).sum())})
        decided = pd.DataFrame(rows, columns=["sec_cik", "convention", "reason", "form_groups", "archive_filings",
                                              "pages_compared", "pages_after_initial_fit_cutoff"])
        staging = output.with_name(f".{output.name}.{uuid4().hex}.pending")
        staging.mkdir(parents=True)
        names = ("issuers.parquet", "pages.parquet", "in_data_counts.parquet")
        try:
            decided.to_parquet(staging / names[0], index=False)
            compared.to_parquet(staging / names[1], index=False)
            counts.to_parquet(staging / names[2], index=False)
            package = Path(__file__).resolve().parents[1]
            by_convention = decided.groupby("convention")
            report: dict[str, Any] = {
                "schema": SCHEMA, "status": "complete", "sec_collection": _collection_pin(root, collection),
                "pages": _collection_pin(root, pages), "source_files": dict(sorted(pins.items())),
                "implementation_files": {f"market_predictor/{name}": file_sha256(package / name)
                                         for name in (*IMPLEMENTATION_PATHS, *RUNNER_PATHS)},
                "artifacts": {name: file_sha256(staging / name) for name in names},
                "conventions": list(CONVENTIONS), "evidence_before_new_york_date": EVIDENCE_BEFORE.isoformat(),
                "rule": ("EDGAR detail pages decide; every form group needs a matching page; in-data counts outside "
                         "EDGAR's weekday 06:00-22:00 New York hours must not contradict; otherwise unknown or fail"),
                "totals": {"issuers_by_convention": {str(key): int(value) for key, value in by_convention.size().items()},
                           "archive_filings_by_convention": {str(key): int(value) for key, value in
                                                             by_convention.archive_filings.sum().items()},
                           "pages_by_state": {str(key): int(value) for key, value in
                                              compared.drop_duplicates("accession_number").page_state.value_counts()
                                              .sort_index().items()},
                           "unknown_issuers": decided.loc[decided.convention.eq(UNKNOWN), ["sec_cik", "reason"]]
                           .to_dict("records")},
                "training_eligible": False, "serving_eligible": False, "promotion_eligible": False,
            }
            write_json_object(staging / "_manifest.json", report)
            _guard()
            staging.rename(output)
        finally:
            if staging.exists():
                for name in (*names, "_manifest.json"):
                    (staging / name).unlink(missing_ok=True)
                staging.rmdir()
        return {**report, "manifest_sha256": file_sha256(output / "_manifest.json")}


def load_sec_acceptance_clock(root: Path, record: dict[str, str], pins: dict[str, str]
                              ) -> tuple[dict[str, str], dict[str, Any]]:
    """The one loader of a published acceptance clock: each issuer's convention and the manifest."""
    manifest = pinned_object(pin_file(root, record, pins), record["sha256"])
    _require(manifest.get("schema") == SCHEMA and manifest.get("status") == "complete", "SEC acceptance clock is not complete")
    path = inside(root, Path(record["path"]).parent / "issuers.parquet")
    _require(file_sha256(path) == manifest["artifacts"]["issuers.parquet"], "SEC acceptance clock differs from its manifest")
    pins[path.relative_to(root).as_posix()] = manifest["artifacts"]["issuers.parquet"]
    issuers = pd.read_parquet(path)
    return dict(zip(issuers.sec_cik.astype(str), issuers.convention.astype(str), strict=True)), manifest
