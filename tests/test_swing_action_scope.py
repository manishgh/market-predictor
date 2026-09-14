"""Exact-record scope review uses synthetic collector receipts, never real payouts."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.sources.http import HttpByteResponse
from market_predictor.sources.official_documents import (
    OfficialDocument,
    collect_official_documents,
    load_official_document_inventory,
    verify_official_document_collection,
)
from market_predictor.swing.contracts.corrected_outcomes import ReviewedCashDistributionScope
from market_predictor.swing.datasets.action_evidence import CorporateActionEvidence
from market_predictor.swing.datasets.action_scope import apply_reviewed_cash_scope, verify_action_scope_documents
from market_predictor.swing.datasets.action_windows import action_window
from market_predictor.swing.datasets.corrected_outcome_admission import reconcile_actions

RECORD = {"id": "qqq-fixture", "symbol": "QQQ", "ex_date": "2023-12-27", "record_date": "2023-12-28",
    "payable_date": "2024-01-16", "special": True, "rate": 0.21584}


@pytest.fixture
def scope(tmp_path: Path) -> ReviewedCashDistributionScope:
    inventory_path = tmp_path / "inventory.toml"
    inventory_path.write_text('''schema_version = "market_predictor.official_document_inventory"
maximum_response_bytes = 1024
attempts_per_document_per_run = 1
[[documents]]
document_id = "fixture_cash_distribution"
url = "https://www.sec.gov/Archives/test-only.htm"
purpose = "Synthetic test-only temporal scope document"
expected_media = "html"
''', encoding="ascii")
    inventory = load_official_document_inventory(inventory_path)
    archive = tmp_path / "archive"

    def fetch(document: OfficialDocument, maximum: int) -> HttpByteResponse:
        body = b"<html>Synthetic scope evidence, not an actual distribution.</html>"
        return HttpByteResponse(body=body, requested_url=document.url, final_url=document.url,
            redirect_chain=(), status_code=200, retrieved_at_utc=datetime.now(UTC), content_type="text/html",
            content_encoding=None, etag=None, last_modified=None, body_length=len(body),
            sha256=hashlib.sha256(body).hexdigest(), body_representation="http_entity_encoded", safe_headers=())

    collect_official_documents(inventory=inventory, output_directory=archive, fetch=fetch)
    report = verify_official_document_collection(archive, inventory)
    receipt_path = archive / report["documents"][0]["attempts"][0]["receipt_path"]
    receipt = json.loads(receipt_path.read_bytes())
    body_path = receipt_path.parent / receipt["body_path"]
    return ReviewedCashDistributionScope.model_validate({
        "action_id": RECORD["id"], "symbol": "QQQ", "family": "cash_dividends",
        "entitlement_basis": "reviewed_cash_ex_distribution_cutoff",
        "record_sha256": json_sha256({"family": "cash_dividends", "record": RECORD}),
        "ex_date": date(2023, 12, 27), "record_date": date(2023, 12, 28), "official_payable_date": date(2024, 1, 15),
        "inventory": {"path": inventory_path.name, "sha256": file_sha256(inventory_path)},
        "archive": "archive", "report_sha256": json_sha256(report), "document_id": "fixture_cash_distribution",
        "document": {"path": body_path.relative_to(tmp_path).as_posix(), "sha256": file_sha256(body_path)},
        "record_locator": "Test-only scope fixture; not a payment observation"})


def evidence(records: list[dict[str, Any]]) -> CorporateActionEvidence:
    by_symbol = {"QQQ": {"cash_dividends": records}}
    digest = json_sha256({"records": by_symbol, "unavailable": [], "source_files": {}, "replay": {}})
    return CorporateActionEvidence("a" * 64, "b" * 64, by_symbol, (), {}, {}, digest)


def test_replayed_scope_does_not_assign_future_rights_or_overwrite_payment(scope: ReviewedCashDistributionScope, tmp_path: Path) -> None:
    pins = verify_action_scope_documents(tmp_path, (scope,))
    assert scope.document.path in pins
    original = evidence([dict(RECORD)])
    projected, gaps = reconcile_actions(original, {"QQQ"}, (scope,))
    window = projected["QQQ"][0]
    assert not gaps and window.first_relevant_date == window.last_relevant_date == date(2023, 12, 27)
    assert window.requires_ownership_before_first_date
    assert not window.intersects(date(2019, 7, 10), date(2019, 7, 23))
    assert window.intersects(date(2023, 12, 20), date(2024, 1, 5))
    assert not window.intersects(date(2024, 1, 17), date(2024, 1, 31))
    assert original.records_by_symbol["QQQ"]["cash_dividends"][0]["payable_date"] == "2024-01-16"
    original.recheck(tmp_path)


def test_unreviewed_special_and_missing_dates_remain_unbounded(scope: ReviewedCashDistributionScope) -> None:
    unreviewed = {**RECORD, "id": "another-action"}
    projected, _ = reconcile_actions(evidence([dict(RECORD), unreviewed]), {"QQQ"}, (scope,))
    assert not projected["QQQ"][0].intersects(date(2019, 7, 10), date(2019, 7, 23))
    assert projected["QQQ"][1].intersects(date(2019, 7, 10), date(2019, 7, 23))
    assert action_window("cash_dividends", {"special": True}).first_relevant_date is None
    assert action_window("spin_offs", {"ex_date": "2023-12-27"}).first_relevant_date is None


@pytest.mark.parametrize("change", [{"rate": 0.0}, {"special": False}, {"ex_date": "2020-01-01"},
    {"symbol": "SPY"}, {"record_date": None}, {"payable_date": "2024-01-15"}])
def test_changed_record_cannot_reuse_reviewed_scope(scope: ReviewedCashDistributionScope, change: dict[str, Any]) -> None:
    with pytest.raises(DataReadinessError, match="exact unresolved provider record"):
        reconcile_actions(evidence([{**RECORD, **change}]), {"QQQ"}, (scope,))


def test_missing_reviewed_action_cannot_silently_skip_scope(scope: ReviewedCashDistributionScope) -> None:
    with pytest.raises(DataReadinessError, match="absent"):
        reconcile_actions(evidence([]), {"QQQ"}, (scope,))


def test_record_date_before_ex_date_cannot_use_reviewed_cash_cutoff(scope: ReviewedCashDistributionScope) -> None:
    record = {**RECORD, "record_date": "2023-12-01"}
    digest = json_sha256({"family": "cash_dividends", "record": record})
    reviewed = scope.model_copy(update={"record_date": date(2023, 12, 1), "record_sha256": digest})
    with pytest.raises(ValidationError, match="due-bill scope"):
        ReviewedCashDistributionScope.model_validate(reviewed.model_dump())
    with pytest.raises(DataReadinessError, match="exact unresolved provider record"):
        apply_reviewed_cash_scope(action_window("cash_dividends", record), record, digest, reviewed)
    assert action_window("cash_dividends", record).intersects(date(2023, 12, 5), date(2023, 12, 18))


@pytest.mark.parametrize("entry,horizon,entitled", [
    (date(2019, 7, 10), date(2019, 7, 23), False),
    (date(2023, 12, 1), date(2023, 12, 26), False),
    (date(2023, 12, 26), date(2023, 12, 27), True),
    (date(2023, 12, 26), date(2024, 1, 31), True),
    (date(2023, 12, 27), date(2024, 1, 10), False),
    (date(2023, 12, 28), date(2024, 1, 15), False),
    (date(2024, 1, 2), date(2024, 1, 16), False),
    (date(2024, 1, 17), date(2024, 1, 31), False),
])
def test_reviewed_ex_distribution_separates_entry_from_payment(scope: ReviewedCashDistributionScope,
    entry: date, horizon: date, entitled: bool) -> None:
    projected, _ = reconcile_actions(evidence([dict(RECORD)]), {"QQQ"}, (scope,))
    assert projected["QQQ"][0].intersects(entry, horizon) is entitled


def test_document_body_tamper_rejected(scope: ReviewedCashDistributionScope, tmp_path: Path) -> None:
    (tmp_path / scope.document.path).write_bytes(b"tampered test-only body")
    with pytest.raises(DataReadinessError):
        verify_action_scope_documents(tmp_path, (scope,))


def test_document_report_external_pin_rejected(scope: ReviewedCashDistributionScope, tmp_path: Path) -> None:
    with pytest.raises(DataReadinessError, match="external pin"):
        verify_action_scope_documents(tmp_path, (scope.model_copy(update={"report_sha256": "0" * 64}),))


def test_document_path_outside_root_rejected(scope: ReviewedCashDistributionScope, tmp_path: Path) -> None:
    document = scope.document.model_copy(update={"path": "../outside.bin"})
    with pytest.raises(DataReadinessError):
        verify_action_scope_documents(tmp_path, (scope.model_copy(update={"document": document}),))


def test_miax_source_allowance_is_limited_to_official_alert_pdfs() -> None:
    payload = {"document_id": "qqq_distribution", "purpose": "Test official source URL validation", "expected_media": "pdf"}
    OfficialDocument(**payload, url="https://www.miaxglobal.com/sites/default/files/alert-files/QQQ_Distribution_53847.pdf")
    for url in ("https://www.miaxglobal.com/private/document.pdf", "https://www.miaxglobal.com/sites/default/files/alert-files/a.pdf?token=x",
            "https://www.miaxglobal.com.evil.example/sites/default/files/alert-files/a.pdf"):
        with pytest.raises(ValidationError):
            OfficialDocument(**payload, url=url)
