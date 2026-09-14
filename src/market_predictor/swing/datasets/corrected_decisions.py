"""Public, lease-bound projection of the complete corrected decision metadata."""
from __future__ import annotations

import tomllib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.sources.official_documents import load_official_document_inventory, verify_official_document_collection
from market_predictor.swing.datasets.corrected_outcomes import (
    _check,
    _guard,
    load_corrected_decision_partition,
    load_corrected_outcome_policy,
)
from market_predictor.swing.datasets.initial_fit_raw_share_plan import verified_initial_fit_raw_share_plan
from market_predictor.swing.datasets.symbol_corrections import pinned_object
from market_predictor.universe.symbol_correction_policy import load_symbol_correction_policy


@dataclass(frozen=True, slots=True)
class CorrectedDecisionProjection:
    """Consume all partitions inside the context; source admission is not conferred."""

    cohort_sha256: str
    retained_security_ids: tuple[str, ...]
    expected_rows: int
    source_files: Mapping[str, str]
    partitions: Iterator[tuple[str, pd.DataFrame]]


def _metadata_implementation(root: Path) -> dict[str, str]:
    package = Path(__file__).resolve().parents[2]
    names = ("swing/datasets/corrected_decisions.py", "swing/datasets/corrected_outcomes.py",
        "swing/datasets/corrected_outcome_admission.py", "swing/contracts/corrected_outcomes.py", "canonical/reconciliation.py")
    return {(package / name).relative_to(root).as_posix(): file_sha256(package / name) for name in names}


@contextmanager
def verified_corrected_decision_partitions(*, root: Path, config: Path, expected_config_sha256: str,
    ) -> Iterator[CorrectedDecisionProjection]:
    """Yield initial-fit metadata only, with corrected and retained parent IDs.

    No action archive or price Parquet is read. The existing raw-plan verifier
    reconstructs the frozen cohort/decision authority from metadata projections.
    The explicit symbol policy and its official document archive are independently
    replayed. Keep downstream writes inside this context so the same lease covers
    their publication and final source rechecks. All partitions must be consumed.
    """
    root = root.resolve()
    config = inside(root, config)
    policy = load_corrected_outcome_policy(root, config, expected_config_sha256)
    selected = pinned_object(inside(root, policy.source_selection.path), policy.source_selection.sha256)
    correction_plan = inside(root, selected["correction_plan"])
    correction_authority = pinned_object(correction_plan / "_authority.json", selected["correction_plan_sha256"])
    correction = pinned_object(correction_plan / "_request.json", correction_authority["request_sha256"])
    parent = inside(root, correction["policy"]["parent_plan"])
    parent_pin = correction["policy"]["parent_plan_sha256"]
    _guard()
    with verified_initial_fit_raw_share_plan(root, inside(root, policy.parent_config.path), parent,
        expected_plan_sha256=parent_pin) as plan:
        authority = pinned_object(parent / "_authority.json", parent_pin)
        request = pinned_object(parent / "_request.json", authority["request_sha256"])
        pins = dict(request["source_files"])
        pins[config.relative_to(root).as_posix()] = expected_config_sha256
        pins.update(_metadata_implementation(root))
        for pin in (policy.source_selection, policy.parent_config, policy.symbol_corrections):
            pins[inside(root, pin.path).relative_to(root).as_posix()] = pin.sha256
        for directory, digest, request_digest in ((parent, parent_pin, authority["request_sha256"]),
                (correction_plan, selected["correction_plan_sha256"], correction_authority["request_sha256"])):
            pins[(directory / "_authority.json").relative_to(root).as_posix()] = digest
            pins[(directory / "_request.json").relative_to(root).as_posix()] = request_digest
        if (len(request["retained_security_ids"]) != 586 or len(request["excluded_security_ids"]) != 45
                or request["decision_start"] != str(policy.decision_start)
                or request["initial_fit_end"] != str(policy.numerical_end)
                or selected["policy_sha256"] != policy.symbol_corrections.sha256):
            raise DataReadinessError("metadata projection differs from frozen cohort, dates or correction policy")
        mapping = load_symbol_correction_policy(root, Path(policy.symbol_corrections.path), policy.symbol_corrections.sha256)
        if mapping.parent_plan_sha256 != parent_pin:
            raise DataReadinessError("decision correction documents belong to another parent plan")
        inventory_path = inside(root, mapping.document_inventory)
        archive = inside(root, mapping.document_archive)
        pins[inventory_path.relative_to(root).as_posix()] = file_sha256(inventory_path)
        pins[(archive / "_request.json").relative_to(root).as_posix()] = file_sha256(archive / "_request.json")
        inventory = load_official_document_inventory(inventory_path)
        documents = verify_official_document_collection(archive, inventory)
        if documents["status"] != "collected_unreviewed" or json_sha256(documents) != mapping.document_report_sha256:
            raise DataReadinessError("corrected decision source documents differ from reviewed archive")
        document_ids = {row["document_id"] for row in documents["documents"]}
        if any(not set(rule.document_ids).issubset(document_ids) for rule in policy.decision_corrections):
            raise DataReadinessError("corrected decision rule lacks reviewed document evidence")
        for document in documents["documents"]:
            for attempt in document["attempts"]:
                receipt_path = inside(archive, attempt["receipt_path"])
                receipt = pinned_object(receipt_path, attempt["receipt_sha256"])
                pins[receipt_path.relative_to(root).as_posix()] = attempt["receipt_sha256"]
                if receipt["body_path"] is not None:
                    body = inside(archive, receipt_path.parent / receipt["body_path"])
                    pins[body.relative_to(root).as_posix()] = receipt["response"]["sha256"]
        parent_policy = tomllib.loads(inside(root, policy.parent_config.path).read_text(encoding="utf-8"))
        preflight_path = inside(root, parent_policy["preflight_path"])
        preflight = pinned_object(preflight_path, pins[preflight_path.relative_to(root).as_posix()])
        manifest_path = inside(root, preflight["request"]["parent_manifest_path"])
        manifest = pinned_object(manifest_path, pins[manifest_path.relative_to(root).as_posix()])
        source: dict[str, Any] = {"manifest_path": manifest_path, "request": request, "source_files": pins}
        expected = int(plan["requirements"]["in_window_decisions"])
        consumed = 0
        complete = False

        def partitions() -> Iterator[tuple[str, pd.DataFrame]]:
            nonlocal consumed, complete
            for record in manifest["files"]:
                if record["first_session"] > str(policy.numerical_end) or record["last_session"] < str(policy.decision_start):
                    continue
                _guard()
                frame = load_corrected_decision_partition(root, source, record, policy)
                consumed += len(frame)
                if not frame.empty:
                    yield record["partition_month"], frame
            complete = True

        _check(root, pins)
        yield CorrectedDecisionProjection(request["cohort_sha256"], tuple(request["retained_security_ids"]),
            expected, MappingProxyType(dict(pins)), partitions())
        if not complete or consumed != expected:
            raise DataReadinessError("corrected decision projection did not consume the complete frozen population")
        _check(root, pins)
