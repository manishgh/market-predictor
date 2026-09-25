"""Leased, immutable inventory of saved SEC filing metadata and saved filing documents."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.sec_filings.collection import (
    _relation_sha256,
    load_sec_filing_collection,
    normalize_sec_identity_relations,
    replay_sec_filing_collection,
)
from market_predictor.catalysts.sec_filings.form_inventory import (
    CURRENT_REPORTS,
    DOCUMENT_STATES,
    SavedDocument,
    filings_frame,
    issuer_inventory,
    saved_document,
    security_years,
)
from market_predictor.config import Settings
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research.issuer_content_inventory import _guard
from market_predictor.research.legacy_query_identity_proofs import pin_file
from market_predictor.sources.official_documents import OfficialDocumentInventory, verify_official_document_collection
from market_predictor.swing.contracts.research_cohort import load_swing_research_cohort
from market_predictor.swing.datasets.initial_fit_issuer_news import LAST_INITIAL_FIT_CUTOFF
from market_predictor.swing.datasets.issuer_news_preparation import FIRST
from market_predictor.swing.datasets.symbol_corrections import pinned_object

CONFIG_SCHEMA = "market_predictor.sec_form_inventory_config"
SCHEMA = "market_predictor.sec_form_inventory"
Mode = Literal["initial_fit", "later_sealed"]
MODES: tuple[Mode, ...] = ("initial_fit", "later_sealed")
IMPLEMENTATION_PATHS = (
    "research/sec_form_inventory.py", "catalysts/sec_filings/form_inventory.py", "catalysts/sec_filings/collection.py",
    "sources/sec.py", "sources/http.py", "sources/official_documents.py", "swing/contracts/research_cohort.py",
    "canonical/store.py", "evidence/hashing.py", "evidence/io.py",
)
_KEYS = frozenset({"schema", "sec_collection", "identity_manifest", "sec_identity_relations", "approved_population",
                   "official_document_collections", "identity_evidence_inventories"})
_EVIDENCE_COLUMNS = (("filing_url", "filing_path", "filing_sha256"), ("evidence_url", "evidence_document", "evidence_raw_sha256"))
_RELATION_KEY = ["security_id", "ticker", "effective_from_utc"]
# A sealed later-window list keeps only what the document collector needs, one row per accession.
SEALED_COLUMNS = ("sec_cik", "accession_number", "sec_form", "item_codes", "report_date", "filing_date", "accepted_at_utc",
                  "primary_document")
SEALED_RULE = "no statistics or reading until qualification rules are frozen on initial-fit evidence"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _records(value: Any, name: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise DataReadinessError(f"SEC form inventory requires a list of {name} pins")
    return value


def _relations(root: Path, settings: dict[str, Any], request: dict[str, Any], pins: dict[str, str]) -> pd.DataFrame:
    identity = pinned_object(pin_file(root, settings["identity_manifest"], pins), settings["identity_manifest"]["sha256"])
    _require(identity.get("schema") == "market_predictor.issuer_news_identity_alignment", "identity alignment manifest differs")
    path = pin_file(root, settings["sec_identity_relations"], pins)
    _require(identity["source_files"].get(path.relative_to(root).as_posix()) == settings["sec_identity_relations"]["sha256"],
             "SEC identity relations differ from the identity alignment manifest's")
    raw = pd.read_parquet(path)
    relations = normalize_sec_identity_relations(raw)
    _require(_relation_sha256(relations) == request.get("identity_relation_sha256"),
             "SEC identity relations differ from the SEC collection request")
    policies = raw.assign(ticker=raw.ticker.astype(str).str.strip().str.upper().str.replace("/", ".", regex=False),
                          effective_from_utc=pd.to_datetime(raw.effective_from_utc, utc=True))[[*_RELATION_KEY, "identity_policy"]]
    joined = relations.merge(policies, on=_RELATION_KEY, how="left", validate="one_to_one")
    _require(bool(joined.identity_policy.notna().all()), "SEC identity relation lacks its identity policy")
    return joined


def _official_documents(root: Path, records: Iterable[dict[str, str]], pins: dict[str, str]
                        ) -> tuple[list[SavedDocument], dict[str, str]]:
    """Replay every official-document collection from its own pinned request with the unchanged verifier."""
    documents: list[SavedDocument] = []
    reports: dict[str, str] = {}
    for record in records:
        path = pin_file(root, record, pins)
        _require(path.name == "_request.json", "official document collections are pinned by their request")
        request = pinned_object(path, record["sha256"])
        inventory = OfficialDocumentInventory.model_validate(request.get("inventory"))
        report = verify_official_document_collection(path.parent, inventory)
        reports[path.parent.relative_to(root).as_posix()] = json_sha256(report)
        urls = {document.document_id: document.url for document in inventory.documents}
        for row in report["documents"]:
            if row["archived"] and (saved := saved_document(urls[row["document_id"]],
                                                            f"official:{path.parent.name}", has_receipt=True)):
                documents.append(saved)
    return documents, reports


def _evidence_documents(root: Path, records: Iterable[dict[str, str]], pins: dict[str, str]) -> list[SavedDocument]:
    """Saved filing documents without HTTP receipts, verified file by file against their pinned inventory."""
    documents: list[SavedDocument] = []
    for record in records:
        path = pin_file(root, record, pins)
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        columns = next((names for names in _EVIDENCE_COLUMNS if set(names) <= set(frame.columns)), None)
        _require(columns is not None, f"identity evidence inventory columns differ: {record['path']}")
        assert columns is not None
        for url, stored, digest in frame.loc[:, list(columns)].itertuples(index=False):
            present = [pd.notna(value) and str(value).strip() != "" for value in (url, stored, digest)]
            if not any(present):
                continue  # A lookup that found no filing saved no document.
            _require(all(present), f"identity evidence row is partly empty: {record['path']}")
            body = inside(root, path.parent.relative_to(root) / Path(str(stored).replace("\\", "/")).name)
            _require(file_sha256(body) == digest, f"identity evidence document hash differs: {body.name}")
            pins[body.relative_to(root).as_posix()] = str(digest)
            if saved := saved_document(str(url), f"identity_evidence:{path.name}", has_receipt=False):
                documents.append(saved)
    return documents


def load_sec_form_inventory(root: Path, record: dict[str, str], pins: dict[str, str], *, allow_sealed_collection: bool = False
                            ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The one loader for published SEC form inventories; a sealed one opens only for document collection."""
    manifest = pinned_object(pin_file(root, record, pins), record["sha256"])
    _require(manifest.get("schema") == SCHEMA and manifest.get("status") == "complete", "SEC form inventory is not complete")
    sealed = bool(manifest.get("sealed_until_rules_frozen"))
    _require(sealed == (manifest.get("mode") == "later_sealed"), "SEC form inventory sealing differs from its mode")
    _require(not sealed or allow_sealed_collection, f"SEC form inventory is sealed: {SEALED_RULE}")
    path = inside(root, Path(record["path"]).parent / "filings.parquet")
    _require(file_sha256(path) == manifest["artifacts"]["filings.parquet"], "SEC form inventory filings differ from its manifest")
    pins[path.relative_to(root).as_posix()] = manifest["artifacts"]["filings.parquet"]
    return pd.read_parquet(path), manifest


def _window(mode: Mode, request: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    if mode == "initial_fit":
        return FIRST, LAST_INITIAL_FIT_CUTOFF
    return LAST_INITIAL_FIT_CUTOFF + pd.Timedelta(1, "ns"), pd.Timestamp(str(request["requested_end_utc"]))


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(series.items())}


def _totals(filings: pd.DataFrame, outside: pd.DataFrame, unrequested: dict[str, int], late: int, cohort: tuple[str, ...],
            relations: pd.DataFrame) -> dict[str, Any]:
    accessions = filings.drop_duplicates("accession_number")
    current = accessions.loc[accessions.sec_form.isin(CURRENT_REPORTS)]
    items = current.assign(item_code=current.item_codes.str.split(",")).explode("item_code")
    return {
        "filing_rows": len(filings), "accessions": len(accessions),
        "securities_with_filings": int(filings.security_id.nunique()),
        "cohort_securities": len(cohort),
        "cohort_securities_without_sec_identity": len(set(cohort) - set(relations.security_id)),
        "accessions_by_form": _counts(accessions.groupby("sec_form").size()),
        "current_report_accessions_by_item": _counts(items.groupby("item_code").size()),
        "current_report_accessions_by_item_and_timing": {f"{item}/{timing}": count for (item, timing), count in
            ((key, int(value)) for key, value in items.groupby(["item_code", "report_timing"]).size().items())},
        "accessions_by_session_position": _counts(accessions.groupby("acceptance_session_position").size()),
        "filing_rows_by_identity_policy": _counts(filings.groupby("identity_policy").size()),
        "share_class_duplicate_rows": int(filings.issuer_cohort_securities.gt(1).sum()),
        "accessions_by_document_status": {state: int(accessions.document_status.eq(state).sum()) for state in DOCUMENT_STATES},
        "cik_identity_outside_relation_rows": len(outside),
        "accepted_in_window_available_after": late,
        "unrequested_form_rows_in_window": dict(sorted(unrequested.items())),
    }


def publish_sec_form_inventory(*, root: Path, config: Path, config_sha256: str, output: Path, mode: Mode) -> dict[str, Any]:
    """Inventory saved SEC metadata and documents for the approved cohort; never content or admission."""
    _require(mode in MODES, f"SEC form inventory mode must be one of {MODES}")
    root = root.resolve()
    config, output = inside(root, config), inside(root, output)
    _require(output.parent == root / "data/research", "SEC form inventory must be a new direct child of data/research")
    runtime = heavy_job_runtime_dir()
    if not runtime.is_absolute():
        runtime = root / runtime
    with heavy_job_lease("inspect-sec-form-inventory", runtime_dir=runtime, config_path=config):
        _guard()
        if output.exists():
            raise FileExistsError(f"immutable SEC form inventory already exists: {output.relative_to(root).as_posix()}")
        pins: dict[str, str] = {}
        relative_config = config.relative_to(root).as_posix()
        settings = pinned_object(pin_file(root, {"path": relative_config, "sha256": config_sha256}, pins), config_sha256)
        _require(set(settings) == _KEYS and settings["schema"] == CONFIG_SCHEMA, "SEC form inventory configuration differs")
        authority = pin_file(root, settings["sec_collection"], pins)
        collection = load_sec_filing_collection(authority.parent)
        _require(authority.name == "_authority.json" and dict(collection.authority) == pinned_object(authority),
                 "SEC collection authority pin differs")
        request = collection.manifest["request"]
        assert isinstance(request, dict)
        relations = _relations(root, settings, request, pins)
        cohort = load_swing_research_cohort(pin_file(root, settings["approved_population"], pins),
                                            source_root=root).retained_security_ids
        official, reports = _official_documents(root, _records(settings["official_document_collections"], "collection"), pins)
        evidence = _evidence_documents(root, _records(settings["identity_evidence_inventories"], "evidence"), pins)
        saved: dict[tuple[str, str], list[SavedDocument]] = {}
        for document in [*official, *evidence]:
            saved.setdefault((document.sec_cik, document.accession_number), []).append(document)
        window = _window(mode, request)
        requested = frozenset(str(form) for form in request["forms"])
        rows: list[dict[str, Any]] = []
        outside_rows: list[dict[str, Any]] = []
        unrequested: dict[str, int] = {}
        late = 0
        for position, issuer in enumerate(replay_sec_filing_collection(collection, Settings())):
            if position % 25 == 0:
                _guard()
            result = issuer_inventory(issuer, requested=requested, relations=relations, cohort=frozenset(cohort),
                                      window=window, saved=saved)
            rows += result.filings
            outside_rows += result.outside_relation
            late += result.available_after_window
            for form, count in result.unrequested_forms.items():
                unrequested[form] = unrequested.get(form, 0) + count
        _guard()
        filings = filings_frame(rows)
        outside = pd.DataFrame(outside_rows, columns=["security_id", "sec_cik", "sec_form", "year_new_york"])
        staging = output.with_name(f".{output.name}.{uuid4().hex}.pending")
        staging.mkdir()
        names = ["filings.parquet"]
        try:
            published = filings if mode == "initial_fit" else filings.loc[:, list(SEALED_COLUMNS)].drop_duplicates(
                "accession_number").reset_index(drop=True)
            published.to_parquet(staging / "filings.parquet", index=False)
            if mode == "initial_fit":
                years, counts = security_years(filings, outside, relations, cohort, window)
                years.to_parquet(staging / "security_years.parquet", index=False)
                counts.to_parquet(staging / "security_year_counts.parquet", index=False)
                names += ["security_years.parquet", "security_year_counts.parquet"]
            package = Path(__file__).resolve().parents[1]
            report: dict[str, Any] = {
                "schema": SCHEMA, "status": "complete", "mode": mode,
                "config": {"path": relative_config, "sha256": config_sha256},
                "window": {"start_utc": window[0].isoformat(), "end_utc": window[1].isoformat(),
                           "rule": "available_at_utc inclusive at both ends"},
                "source_files": dict(sorted(pins.items())), "document_collection_reports": dict(sorted(reports.items())),
                "implementation_files": {f"market_predictor/{name}": file_sha256(package / name) for name in IMPLEMENTATION_PATHS},
                "artifacts": {name: file_sha256(staging / name) for name in names},
                "availability_basis": "sec_daily_swing_conservative_proxy; first observed 2026, retrospective research only",
                "count_scope": "EDGAR form names and 8-K item codes; not event meaning or content qualification",
                "zero_scope": "zero is verified for requested forms only inside an SEC identity relation",
                "training_eligible": False, "serving_eligible": False, "promotion_eligible": False,
            }
            if mode == "initial_fit":
                report["totals"] = _totals(filings, outside, unrequested, late, cohort, relations)
            else:
                report.update(sealed_until_rules_frozen=True, accessions=len(published), reading_rule=SEALED_RULE)
            write_json_object(staging / "_manifest.json", report)
            _guard()
            staging.rename(output)
        finally:
            if staging.exists():
                for name in (*names, "_manifest.json"):
                    (staging / name).unlink(missing_ok=True)
                staging.rmdir()
        return {**report, "manifest_sha256": file_sha256(output / "_manifest.json")}
