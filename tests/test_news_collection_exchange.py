"""Collector-generated cross-language test bytes, never historical market evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from market_predictor.evidence.news_collection import (
    NewsCollectionPlan,
    collect_news_receipts,
    news_collection_plan_bytes,
    news_collection_plan_sha256,
)
from market_predictor.evidence.news_exchange import validate_news_receipt


def test_collection_exchange_fixture_matches_actual_publication(tmp_path: Path) -> None:
    vector = json.loads((Path(__file__).parent / "fixtures/news_collection_exchange.json").read_text(encoding="utf-8"))
    assert vector["purpose"] == "synthetic_test_only_not_market_evidence"
    plan = NewsCollectionPlan.model_validate_json(vector["plan_utf8"])
    assert news_collection_plan_bytes(plan) == vector["plan_utf8"].encode()
    assert news_collection_plan_sha256(plan) == vector["plan_sha256"]
    receipt = validate_news_receipt(vector["manifest_utf8"].encode(), vector["payload_utf8"].encode())
    report = collect_news_receipts(root=tmp_path, plan=plan, expected_plan_sha256=vector["plan_sha256"],
        fetch_page=lambda request: receipt)
    assert report.complete
    attempt = tmp_path / "runs" / vector["plan_sha256"] / "attempts/MSFT/000001"
    for path, key in (("intent.json", "intent_utf8"), ("result/result.json", "result_utf8"),
        ("receipt/manifest.json", "manifest_utf8"), ("receipt/payload.json", "payload_utf8")):
        assert (attempt / path).read_bytes() == vector[key].encode()
    result = json.loads(vector["result_utf8"])
    assert result["intent_sha256"] == hashlib.sha256(vector["intent_utf8"].encode()).hexdigest()
    assert result["receipt_sha256"] == report.windows[0].receipts[0].receipt_sha256
    assert not receipt.observable_at(receipt.manifest.request.end_utc)
    assert collect_news_receipts(root=tmp_path, plan=plan, expected_plan_sha256=vector["plan_sha256"]) == report
