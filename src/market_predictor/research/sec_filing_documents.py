"""Leased, resumable collection of selected SEC filing documents from a pinned SEC form inventory."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.catalysts.sec_filings import document_collection as documents
from market_predictor.config import Settings
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.io import inside
from market_predictor.research.sec_form_inventory import SEALED_RULE, load_sec_form_inventory
from market_predictor.research.sec_page_collection import open_collection, run_collection

SCHEMA = "market_predictor.sec_filing_document_collection"
IMPLEMENTATION_PATHS = ("research/sec_filing_documents.py",)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def open_sec_document_collection(root: Path, record: dict[str, str]) -> tuple[documents.Store, dict[str, Any]]:
    """The one reader of a completed collection; a sealed one stays closed until qualification rules are frozen."""
    store, manifest = open_collection(root, record, SCHEMA)
    _require(not manifest.get("sealed_until_rules_frozen"), f"SEC document collection is sealed: {SEALED_RULE}")
    return store, manifest


def collect_sec_filing_documents(*, root: Path, inventory: dict[str, str], output: Path,
                                 resume_checkpoint_sha256: str | None = None, pilot_accessions: tuple[str, ...] = (),
                                 settings: Settings | None = None) -> dict[str, Any]:
    """Collect the detail page, primary document and EX-99 exhibits of every selected filing; never content review."""
    accessions = tuple(sorted(pilot_accessions))

    def plan(pins: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
        resolved = root.resolve()
        filings, manifest = load_sec_form_inventory(resolved, inventory, pins, allow_sealed_collection=True)
        units = documents.work_list(filings)
        if accessions:
            missing = set(accessions) - set(units.accession_number)
            _require(not missing, f"pilot accessions are not selected filings: {sorted(missing)}")
            units = units.loc[units.accession_number.isin(accessions)].reset_index(drop=True)
        return units, {
            "inventory": {"path": inside(resolved, inventory["path"]).relative_to(resolved).as_posix(),
                          "sha256": inventory["sha256"]},
            "inventory_mode": manifest["mode"], "sealed_until_rules_frozen": bool(manifest.get("sealed_until_rules_frozen")),
            "pilot_accessions": list(accessions),
            "selection": {"forms": list(documents.CURRENT_REPORTS), "items": sorted(documents.SELECTED_ITEMS),
                          "documents": "primary document and every EX-99 exhibit named by a verified detail page"},
        }

    return run_collection(root=root, output=output, job="collect-sec-filing-documents", schema=SCHEMA, plan=plan,
                          phases=("index", "document"), implementation_paths=IMPLEMENTATION_PATHS,
                          resume_checkpoint_sha256=resume_checkpoint_sha256, settings=settings)
