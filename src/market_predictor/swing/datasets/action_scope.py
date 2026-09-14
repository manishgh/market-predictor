"""Independently replayed, exact-record temporal bounds for reviewed cash actions."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.sources.official_documents import load_official_document_inventory, verify_official_document_collection
from market_predictor.swing.contracts.corrected_outcomes import ReviewedCashDistributionScope
from market_predictor.swing.datasets.action_windows import ActionWindow
from market_predictor.swing.datasets.symbol_corrections import pinned_object


def verify_action_scope_documents(root: Path, scopes: tuple[ReviewedCashDistributionScope, ...]) -> dict[str, str]:
    """Return immutable source pins; the caller's publication lease must be held."""
    pins: dict[str, str] = {}
    reports: dict[tuple[str, str, str], dict[str, Any]] = {}
    for scope in scopes:
        assert_system_memory_available(minimum_available_gib=2.0)
        inventory_path = inside(root, scope.inventory.path)
        archive = inside(root, scope.archive)
        key = (scope.archive, scope.inventory.sha256, scope.report_sha256)
        if key not in reports:
            if file_sha256(inventory_path) != scope.inventory.sha256:
                raise DataReadinessError("reviewed action scope inventory changed")
            pins[inventory_path.relative_to(root).as_posix()] = scope.inventory.sha256
            request = archive / "_request.json"
            pins[request.relative_to(root).as_posix()] = file_sha256(request)
            report = verify_official_document_collection(archive, load_official_document_inventory(inventory_path))
            if report["status"] != "collected_unreviewed" or json_sha256(report) != scope.report_sha256:
                raise DataReadinessError("reviewed action scope document replay differs from external pin")
            reports[key] = report
        document_path = inside(root, scope.document.path)
        found = False
        for document in reports[key]["documents"]:
            for attempt in document["attempts"]:
                receipt_path = inside(archive, attempt["receipt_path"])
                receipt = pinned_object(receipt_path, attempt["receipt_sha256"])
                pins[receipt_path.relative_to(root).as_posix()] = attempt["receipt_sha256"]
                if receipt["body_path"] is None:
                    continue
                body = inside(archive, receipt_path.parent / receipt["body_path"])
                digest = receipt["response"]["sha256"]
                pins[body.relative_to(root).as_posix()] = digest
                found |= (document["document_id"] == scope.document_id and body == document_path
                    and digest == scope.document.sha256 and receipt["state"] == "archived_unreviewed")
        if not found:
            raise DataReadinessError("reviewed action scope body lacks independently replayed receipt")
    for name, digest in pins.items():
        if file_sha256(inside(root, name)) != digest:
            raise DataReadinessError("reviewed action scope source changed during replay")
    return pins


def apply_reviewed_cash_scope(window: ActionWindow, record: dict[str, Any], digest: str,
    scope: ReviewedCashDistributionScope) -> ActionWindow:
    """Identify entitled holdings without supplying their missing payment or value."""
    if (window.family != scope.family or window.action_id != scope.action_id or digest != scope.record_sha256
            or record.get("symbol") != scope.symbol or record.get("special") is not True
            or record.get("ex_date") != scope.ex_date.isoformat() or record.get("record_date") != scope.record_date.isoformat()
            or scope.record_date < scope.ex_date or scope.official_payable_date < scope.ex_date
            or window.unavailable_reason != "unbounded_entitlement_or_due_bill_dates"):
        raise DataReadinessError("reviewed cash scope does not match the exact unresolved provider record")
    # The reviewed ex-distribution cutoff separates entitlement from payment.
    # Newly opened ex-date shares do not inherit an earlier owner's unpaid claim.
    return replace(window, first_relevant_date=scope.ex_date, last_relevant_date=scope.ex_date,
        requires_ownership_before_first_date=True, unavailable_reason="reviewed_cash_distribution_resolution_unavailable")
