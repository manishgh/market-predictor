"""Saved Alpaca news authorities written by the real collector and derivation (test-only)."""
from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from market_predictor.canonical.audits import CanonicalAuditReport, audit_universe_memberships
from market_predictor.canonical.normalize import canonicalize_universe_memberships
from market_predictor.canonical.store import file_sha256, load_canonical_artifact, write_canonical_artifact
from market_predictor.catalysts.issuer_events.alpaca_news_collection import collect_alpaca_news_history
from market_predictor.catalysts.issuer_events.attribution import ATTRIBUTION_POLICY_SHA256, ATTRIBUTION_POLICY_VERSION
from market_predictor.catalysts.issuer_events.attribution_history import ATTRIBUTION_SCOPE_POLICY
from market_predictor.evidence.hashing import json_sha256
from market_predictor.sources.alpaca import AlpacaNewsPage
from market_predictor.swing.datasets.initial_fit_issuer_news import (
    FINBERT_REVISION,
    SavedIssuerAuthority,
    _write_frame,
    derive_initial_fit_issuer_inputs,
)
from tests.test_swing_catalyst_lineage import _relations, _sentiments

News = Callable[[str, datetime, datetime], list[list[dict[str, Any]]]]


def article(identifier: int, created: str, *, symbols: list[str], updated: str | None = None,
            **changes: Any) -> dict[str, Any]:
    return {"id": identifier, "headline": f"Report {identifier}", "content": "Body", "summary": "Summary",
            "created_at": created, "updated_at": updated, "source": "benzinga",
            "url": f"https://example.test/{identifier}", "symbols": symbols, **changes}


def _json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")


def collect(root: Path, monkeypatch: pytest.MonkeyPatch, name: str, securities: list[tuple[str, str]], news: News, *,
            start: date, end: date, chunk_days: int) -> str:
    """Write one complete raw archive with repository-relative paths, as the real runs did."""
    monkeypatch.chdir(root)
    raw = pd.DataFrame({"ticker": [ticker for ticker, _ in securities], "security_id": [security for _, security in securities],
        "effective_from_utc": pd.Timestamp("2019-01-01T00:00:00Z"), "effective_to_utc": pd.NaT,
        "available_at_utc": pd.Timestamp("2019-01-01T00:00:00Z"), "sector": "Technology", "industry": "Software",
        "market_cap_bucket": "large", "liquidity_bucket": "high", "primary_benchmark": "XLK",
        "universe_snapshot_id": "test-memberships", "source": "test", "availability_policy": "provider_publication_proxy"})
    memberships = canonicalize_universe_memberships(raw)
    path = Path(f"data/memberships/{name}.parquet")
    write_canonical_artifact(memberships, path, artifact_type="memberships", production_ready=False,
        audit=CanonicalAuditReport(checks=audit_universe_memberships(memberships, require_observed=False)))

    def fetch(symbol: str, window_start: datetime, window_end: datetime, token: str | None) -> AlpacaNewsPage:
        pages = news(symbol, window_start, window_end) or [[]]
        index = 0 if token is None else int(token)
        return AlpacaNewsPage(request_page_token=token, news=tuple(pages[index]),
                              next_page_token=str(index + 1) if index < len(pages) - 1 else None)

    result = collect_alpaca_news_history(memberships_path=path, start_date=start, end_date=end,
        out_dir=Path(f"data/raw/{name}"), fetch_page=fetch, provider_symbol_for=lambda ticker: ticker.replace("-", "."),
        workers=1, chunk_days=chunk_days)
    assert result.status == "complete"
    return f"data/raw/{name}"


def saved_authority(root: Path, collection: str, name: str, *, blindspots: tuple[str, ...] = (),
                    corrected: bool = False) -> dict[str, Any]:
    """Strict attribution, FinBERT scores and audit over a real collection, one direct relation per event."""
    manifest = json.loads((root / collection / "_manifest.json").read_text(encoding="utf-8"))
    base = root / "data/research" / name
    labels = _write_frame(pd.DataFrame({"company": ["Synthetic Only"]}), base / "labels.parquet", "security_business_labels", {})
    identities = _write_frame(pd.DataFrame({"company": ["Synthetic Only"], "security_id": ["security:synthetic"],
        "ticker": ["SYN"], "effective_from_utc": pd.to_datetime(["2019-01-01T00:00:00Z"], utc=True),
        "effective_to_utc": pd.to_datetime(["2026-07-09T00:00:00Z"], utc=True),
        "available_at_utc": pd.to_datetime(["2019-01-01T00:00:00Z"], utc=True)}),
        base / "identities.parquet", "security_business_label_coverage", {})
    audit = base / "audit.json"
    _json(audit, {"passed": True, "request_sha256": manifest["request_sha256"], "coverage_blindspot_security_ids": list(blindspots)})
    excluded = [] if corrected else list(blindspots)
    common: dict[str, Any] = {"collection_manifest_path": f"{collection}/_manifest.json",
        "collection_manifest_sha256": file_sha256(root / collection / "_manifest.json"),
        "collection_audit_path": audit.relative_to(root).as_posix(), "collection_audit_sha256": file_sha256(audit),
        "collection_request_sha256": manifest["request_sha256"], "excluded_security_ids": excluded, "production_ready": False}
    scope: dict[str, Any] = {"scope_policy": ATTRIBUTION_SCOPE_POLICY, "source_coverage_admitted": False} if corrected else {}
    requests = {"attribution": {**common, **scope, "schema": "swing.event_attribution_request.v1",
        "attribution_policy_version": ATTRIBUTION_POLICY_VERSION, "attribution_policy_sha256": ATTRIBUTION_POLICY_SHA256,
        "business_labels_path": labels["path"], "business_labels_sha256": labels["sha256"],
        "security_identities_path": identities["path"], "security_identities_sha256": identities["sha256"]},
        "sentiment": {**common, **scope, "schema": "swing.event_sentiment_request.v1", "model_name": "ProsusAI/finbert",
            "model_revision": FINBERT_REVISION}}
    digests = {kind: json_sha256(request) for kind, request in requests.items()}
    for kind, request in requests.items():
        _json(base / kind / "_request.json", {**request, "request_sha256": digests[kind]})
    rows: dict[str, list[dict[str, Any]]] = {"attribution": [], "sentiment": []}
    for record in manifest["artifacts"]:
        if record["security_id"] in blindspots:
            continue
        events, _ = load_canonical_artifact(root / record["path"], allow_research=True)
        available = pd.to_datetime(events.feature_available_at_utc, utc=True)
        relations = pd.concat([_relations(time) for time in available], ignore_index=True).assign(
            event_id=events.event_id.to_numpy(), relation_id=(events.event_id + "-relation").to_numpy(),
            event_feature_available_at_utc=available.to_numpy(), source_security_id=events.security_id.to_numpy(),
            target_security_id=events.security_id.to_numpy(), source_ticker=events.ticker.to_numpy(),
            target_ticker=events.ticker.to_numpy())
        scores = pd.concat([_sentiments().iloc[[0]]] * len(events), ignore_index=True).assign(
            event_id=events.event_id.to_numpy(), security_id=events.security_id.to_numpy(), ticker=events.ticker.to_numpy(),
            published_at_utc=available.to_numpy(), event_available_at_utc=available.to_numpy(),
            research_feature_available_at_utc=(available + pd.Timedelta(minutes=5)).to_numpy(),
            sentiment_model_revision=FINBERT_REVISION, sentiment_input_sha256=[json_sha256({"event": e}) for e in events.event_id])
        inputs = {"source_event_artifact_sha256": record["sha256"], "chunk_id": record["chunk_id"]}
        for kind, frame, folder, canonical_kind, key in (
                ("attribution", relations, "relations", "event_security_relations", "event_attribution_request_sha256"),
                ("sentiment", scores, "sentiment", "event_sentiment_research", "sentiment_request_sha256")):
            written = _write_frame(frame, base / kind / folder / f"{record['chunk_id']}.parquet", canonical_kind,
                                   {**inputs, key: digests[kind]})
            rows[kind].append({**written, "chunk_id": record["chunk_id"]})
    for kind, schema in (("attribution", "swing.event_attribution_manifest.v1"), ("sentiment", "swing.event_sentiment_manifest.v1")):
        extra = {**scope, "coverage_blindspot_security_ids": list(blindspots)} if corrected else {}
        _json(base / kind / "_manifest.json", {"schema": schema, "status": "complete", "production_ready": False,
            "failed_chunks": {}, "request_sha256": digests[kind], "excluded_security_ids": excluded, "artifacts": rows[kind], **extra})
    return {"base": base, "audit": audit, "labels": labels, "identities": identities}


def derive(root: Path, authority: dict[str, Any], name: str) -> dict[str, Any]:
    base = authority["base"]
    source = SavedIssuerAuthority(base / "attribution", file_sha256(base / "attribution/_manifest.json"),
        base / "sentiment", file_sha256(base / "sentiment/_manifest.json"),
        file_sha256(Path(authority["labels"]["path"] + ".manifest.json")),
        file_sha256(Path(authority["identities"]["path"] + ".manifest.json")))
    with patch("market_predictor.swing.datasets.initial_fit_issuer_news.assert_system_memory_available"):
        derived = derive_initial_fit_issuer_inputs(source=source, output_directory=root / "data/research" / name,
                                                   repository_root=root)
    return {"kind": "derived", "directory": f"data/research/{name}", "manifest_sha256": derived.manifest_sha256}


def corrected_spec(root: Path, collection: str, authority: dict[str, Any]) -> dict[str, Any]:
    base = authority["base"]
    return {"kind": "corrected",
        **{kind: {"directory": directory, "manifest_sha256": file_sha256(root / directory / "_manifest.json")} for kind, directory in
           (("collection", collection), ("attribution", (base / "attribution").relative_to(root).as_posix()),
            ("sentiment", (base / "sentiment").relative_to(root).as_posix()))},
        "audit": {"path": authority["audit"].relative_to(root).as_posix(), "sha256": file_sha256(authority["audit"])}}
