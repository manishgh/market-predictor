"""Test-only synthetic query evidence; no fixture establishes issuer membership."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
from pydantic import ValidationError

from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.catalysts.issuer_events.alpaca_news_audit import (
    audit_alpaca_news_history,
)
from market_predictor.catalysts.issuer_events.alpaca_news_collection import (
    NewsHistoryCollectionResult,
    NewsPageFetcher,
    collect_alpaca_news_history,
)
from market_predictor.catalysts.issuer_events.news_query_scope import load_news_query_scope
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.sources.alpaca import AlpacaNewsPage

_CUTOFF = datetime(2023, 6, 7, 13, 30, tzinfo=UTC)
_JUST_BEFORE = "2023-06-07T13:29:59.999999Z"
_SECURITY_ID = "test-only:synthetic-fisv"


class IssuerNewsQueryScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.out_dir = self.root / "news"
        self.source = self.root / "synthetic-source.txt"
        self.source.write_text(
            "SYNTHETIC TEST-ONLY source pin. Not an official issuer source, "
            "index membership, or production evidence.\n",
            encoding="utf-8",
        )
        self.scope_path = self.root / "synthetic-query-scope.json"
        self.scope: dict[str, Any] = {
            "schema": "market_predictor.issuer_news_query_scope",
            "purpose": "historical_source_collection_only",
            "production_ready": False,
            "intervals": [self._interval()],
        }
        self._save_scope()

    def _interval(self, **overrides: Any) -> dict[str, Any]:
        return {
            "ticker": "FISV",
            "security_id": _SECURITY_ID,
            "start_utc": "2023-06-01T00:00:00Z",
            "end_exclusive_utc": "2023-06-07T13:30:00Z",
            "interpretation": "SYNTHETIC TEST ONLY: half-open provider query, not membership",
            "source_files": [
                {"path": self.source.name, "sha256": file_sha256(self.source)},
            ],
            **overrides,
        }

    def _save_scope(self) -> None:
        _write_json(self.scope_path, self.scope)

    def _collect(
        self,
        fetch_page: NewsPageFetcher,
        *,
        guard: Callable[[], object] | None = None,
    ) -> NewsHistoryCollectionResult:
        return collect_alpaca_news_history(
            query_scope_path=self.scope_path,
            query_scope_root=self.root,
            start_date=date(2023, 5, 31),
            end_date=date(2023, 6, 7),
            out_dir=self.out_dir,
            fetch_page=fetch_page,
            provider_symbol_for=lambda ticker: ticker,
            workers=1,
            chunk_days=7,
            system_memory_guard=guard,
        )

    def test_requires_exactly_one_scope_input_before_fetching(self) -> None:
        for both in (False, True):
            with self.subTest(both=both):
                fetch = Mock(side_effect=AssertionError("must not fetch"))
                with self.assertRaisesRegex(DataReadinessError, "exactly one"):
                    collect_alpaca_news_history(
                        memberships_path=self.root / "nonexistent.parquet" if both else None,
                        query_scope_path=self.scope_path if both else None,
                        query_scope_root=self.root,
                        start_date=date(2023, 6, 1),
                        end_date=date(2023, 6, 7),
                        out_dir=self.out_dir,
                        fetch_page=fetch,
                        provider_symbol_for=lambda ticker: ticker,
                        workers=1,
                    )
                fetch.assert_not_called()
                self.assertFalse(self.out_dir.exists())

    def test_rejects_naive_empty_and_reversed_intervals(self) -> None:
        cases = (
            ({"start_utc": "2023-06-01T00:00:00"}, "timezone"),
            ({"end_exclusive_utc": "2023-06-07T13:30:00"}, "timezone"),
            ({"end_exclusive_utc": "2023-06-01T00:00:00Z"}, "empty or reversed"),
            ({"start_utc": "2023-06-08T00:00:00Z"}, "empty or reversed"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                self.scope["intervals"] = [self._interval(**overrides)]
                self._save_scope()
                with self.assertRaisesRegex(ValidationError, message):
                    load_news_query_scope(self.scope_path, root=self.root)

    def test_rejects_overlaps_by_ticker_or_security_identity(self) -> None:
        cases = (
            {},
            {"security_id": "test-only:other-security"},
            {"ticker": "FI"},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                self.scope["intervals"] = [
                    self._interval(),
                    self._interval(start_utc="2023-06-06T00:00:00Z", **overrides),
                ]
                self._save_scope()
                with self.assertRaisesRegex(DataReadinessError, "overlapping or ambiguous"):
                    load_news_query_scope(self.scope_path, root=self.root)

    def test_adjacent_intervals_are_queries_not_memberships(self) -> None:
        self.scope["intervals"].append(
            self._interval(
                ticker="FI",
                start_utc="2023-06-07T13:30:00Z",
                end_exclusive_utc="2023-06-08T00:00:00Z",
            )
        )
        self._save_scope()
        intervals, identity = load_news_query_scope(self.scope_path, root=self.root)
        self.assertEqual(intervals["ticker"].tolist(), ["FISV", "FI"])
        self.assertEqual(intervals["security_id"].tolist(), [_SECURITY_ID] * 2)
        self.assertEqual(intervals.loc[0, "effective_to_utc"], _CUTOFF)
        self.assertEqual(intervals.loc[1, "effective_from_utc"], _CUTOFF)
        self.assertEqual(
            identity["scope_kind"], "explicit_issuer_query_intervals_not_index_membership"
        )
        self.assertEqual(identity["query_scope_sha256"], file_sha256(self.scope_path))
        self.assertEqual(identity["query_source_files"], {self.source.name: file_sha256(self.source)})
        self.assertNotIn("memberships_path", identity)
        self.assertNotIn("memberships_sha256", identity)

    def test_scope_cannot_claim_production_readiness(self) -> None:
        self.scope["production_ready"] = True
        self._save_scope()
        with self.assertRaisesRegex(ValidationError, "production_ready"):
            load_news_query_scope(self.scope_path, root=self.root)

    def test_source_pin_tamper_rejected_before_network_launch(self) -> None:
        self.source.write_text("SYNTHETIC TEST-ONLY altered source.\n", encoding="utf-8")
        fetch = Mock(side_effect=AssertionError("must not fetch"))
        with self.assertRaisesRegex(DataReadinessError, "source pin differs"):
            load_news_query_scope(self.scope_path, root=self.root)
        with self.assertRaisesRegex(DataReadinessError, "source pin differs"):
            self._collect(fetch)
        fetch.assert_not_called()
        self.assertFalse(self.out_dir.exists())

    def test_date_envelope_cannot_silently_clip_explicit_scope(self) -> None:
        fetch = Mock(side_effect=AssertionError("must not fetch"))
        with self.assertRaisesRegex(DataReadinessError, "date envelope"):
            collect_alpaca_news_history(
                query_scope_path=self.scope_path,
                query_scope_root=self.root,
                start_date=date(2023, 6, 2),
                end_date=date(2023, 6, 7),
                out_dir=self.out_dir,
                fetch_page=fetch,
                provider_symbol_for=lambda ticker: ticker,
                workers=1,
            )
        fetch.assert_not_called()

    def test_shared_collector_filters_exact_fisv_cutoff_and_replays_pages(self) -> None:
        calls: list[tuple[str, datetime, datetime, str | None]] = []

        def fetch(symbol: str, start: datetime, end: datetime, token: str | None) -> AlpacaNewsPage:
            calls.append((symbol, start, end, token))
            if token is None:
                return AlpacaNewsPage(None, "synthetic-page-2", (_news(1, _JUST_BEFORE),))
            return AlpacaNewsPage(
                token,
                None,
                (
                    _news(1, _JUST_BEFORE),
                    _news(2, "2023-06-07T13:30:00Z"),
                    _news(3, _JUST_BEFORE, ticker="FI"),
                ),
            )

        result = self._collect(fetch)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.requested_chunks, 1)
        self.assertEqual(result.observed_chunks, 1)
        start = datetime(2023, 6, 1, tzinfo=UTC)
        self.assertEqual(calls, [("FISV", start, _CUTOFF, None), ("FISV", start, _CUTOFF, "synthetic-page-2")])
        request = _read_json(self.out_dir / "_request.json")
        self.assertNotIn("memberships_path", request)
        self.assertNotIn("memberships_sha256", request)
        self.assertEqual(request["scope_kind"], "explicit_issuer_query_intervals_not_index_membership")
        self.assertEqual(request["query_scope_sha256"], file_sha256(self.scope_path))
        self.assertIs(request["production_ready"], False)
        manifest = _read_json(self.out_dir / "_manifest.json")
        self.assertIs(manifest["production_ready"], False)
        self.assertEqual(manifest["total_rows"], 1)
        artifact = manifest["artifacts"][0]
        self.assertIs(artifact["production_ready"], False)
        events, event_manifest = load_canonical_artifact(
            Path(artifact["path"]), expected_type="events", allow_research=True
        )
        self.assertIs(event_manifest["production_ready"], False)
        self.assertEqual(events["security_id"].tolist(), [_SECURITY_ID])
        self.assertEqual(events["ticker"].tolist(), ["FISV"])
        self.assertEqual(events["published_at_utc"].tolist(), [pd.Timestamp(_JUST_BEFORE)])
        self.assertEqual(events["availability_policy"].tolist(), ["provider_publication_proxy"])
        ledger, ledger_manifest = load_canonical_artifact(
            self.out_dir / "_source_collections.parquet",
            expected_type="source_collections",
            allow_research=True,
        )
        self.assertIs(ledger_manifest["production_ready"], False)
        for key, value in {
            "pages": 2, "provider_rows": 4, "accepted_rows": 1,
            "outside_window_rows": 1, "symbol_mismatch_rows": 1, "duplicate_rows": 1,
        }.items():
            self.assertEqual(int(ledger.loc[0, key]), value, key)
        report, summary = audit_alpaca_news_history(self.out_dir)
        self.assertTrue(summary["passed"], summary["errors"])
        self.assertEqual(summary["page_count"], 2)
        self.assertEqual(summary["event_rows"], 1)
        self.assertEqual(report.loc[0, "audit_errors"], "")

    def test_audit_rejects_source_pin_tamper(self) -> None:
        self.assertEqual(self._collect(_empty_page).status, "complete")
        self.source.write_text("SYNTHETIC TEST-ONLY altered after collection.\n", encoding="utf-8")
        with self.assertRaisesRegex(DataReadinessError, "source pin differs"):
            audit_alpaca_news_history(self.out_dir)

    def test_audit_rejects_rehashed_shifted_or_omitted_scope_chunks(self) -> None:
        self.scope["intervals"][0]["start_utc"] = "2023-05-31T00:00:00Z"
        self._save_scope()
        self.assertEqual(self._collect(_empty_page).requested_chunks, 2)
        _, summary = audit_alpaca_news_history(self.out_dir)
        self.assertTrue(summary["passed"], summary["errors"])
        request_path = self.out_dir / "_request.json"
        original = request_path.read_text(encoding="utf-8")
        for mutation in ("shifted", "omitted"):
            with self.subTest(mutation=mutation):
                request = json.loads(original)
                if mutation == "shifted":
                    request["work_units"][0]["start_utc"] = "2023-05-31T00:00:01+00:00"
                else:
                    request["work_units"].pop()
                # A valid self-hash must not replace reconstruction from the pinned scope.
                del request["request_sha256"]
                request["request_sha256"] = hashlib.sha256(
                    json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
                ).hexdigest()
                _write_json(request_path, request)
                with self.assertRaisesRegex(DataReadinessError, "work units differ from source scope"):
                    audit_alpaca_news_history(self.out_dir)

    def test_memory_pressure_preserves_first_page_stops_launches_and_resumes(self) -> None:
        self.scope["intervals"][0]["start_utc"] = "2023-05-31T00:00:00Z"
        self._save_scope()
        for next_token in ("synthetic-page-2", None):
            with self.subTest(next_token=next_token):
                self._assert_pressure_resume(next_token)

    def test_invalid_replay_chunk_size_fails_before_reconstruction(self) -> None:
        self._collect(_empty_page)
        path = self.out_dir / "_request.json"
        original = _read_json(path)
        for value in (0, -1, 1.5, True, 367):
            for rehash in (False, True):
                with self.subTest(value=value, rehash=rehash):
                    request = {**original, "chunk_days": value}
                    if rehash:
                        del request["request_sha256"]
                        request["request_sha256"] = hashlib.sha256(json.dumps(request,
                            sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
                    _write_json(path, request)
                    with self.assertRaisesRegex(DataReadinessError, "chunk_days|request identity"):
                        audit_alpaca_news_history(self.out_dir)

    def _assert_pressure_resume(self, next_token: str | None) -> None:
        first_start = datetime(2023, 5, 31, tzinfo=UTC)
        second_start = datetime(2023, 6, 7, tzinfo=UTC)
        self.out_dir = self.root / ("nonterminal" if next_token else "terminal")
        calls: list[tuple[datetime, str | None]] = []
        pressure_pages: list[Path] = []

        def fetch(symbol: str, start: datetime, end: datetime, token: str | None) -> AlpacaNewsPage:
            self.assertEqual(symbol, "FISV")
            calls.append((start, token))
            return AlpacaNewsPage(token, next_token, (_news(10, start.isoformat()),))

        def guard() -> None:
            saved = sorted((self.out_dir / "raw_pages").rglob("page_*.json"))
            if saved:
                pressure_pages.extend(saved)
                raise MemoryBudgetError("synthetic test-only system memory pressure")

        with self.assertRaisesRegex(MemoryBudgetError, "persisted pages/attempts can be resumed"):
            self._collect(fetch, guard=guard)
        self.assertEqual(calls, [(first_start, None)])
        pages = list((self.out_dir / "raw_pages").rglob("page_*.json"))
        self.assertEqual(len(pages), 1)
        self.assertEqual(pressure_pages, pages)
        saved_path = pages[0]
        saved_bytes = saved_path.read_bytes()
        self.assertEqual(_read_json(saved_path)["next_page_token"], next_token)
        self.assertFalse((self.out_dir / "_manifest.json").exists())
        self.assertFalse((self.out_dir / "_status.json").exists())
        self.assertFalse((self.out_dir / "_source_collections.parquet").exists())
        self.assertEqual(list((self.out_dir / "events").glob("*.parquet")), [])
        attempts = list((self.out_dir / "attempts").glob("*.json"))
        self.assertEqual(len(attempts), 1)
        attempt = _read_json(attempts[0])
        self.assertEqual(attempt["collection"]["status"], "failed")
        self.assertEqual(attempt["collection"]["error_type"], "MemoryBudgetError")
        resume_calls: list[tuple[datetime, str | None]] = []

        def resume(symbol: str, start: datetime, end: datetime, token: str | None) -> AlpacaNewsPage:
            resume_calls.append((start, token))
            if start == first_start:
                self.assertEqual(token, "synthetic-page-2")
                return AlpacaNewsPage(token, None, ())
            self.assertEqual(start, second_start)
            self.assertEqual(end, _CUTOFF)
            self.assertIsNone(token)
            return AlpacaNewsPage(None, None, (_news(11, start.isoformat()),))

        healthy_guard = Mock(return_value=None)
        result = self._collect(resume, guard=healthy_guard)
        expected_calls = [(first_start, next_token)] if next_token else []
        self.assertEqual(resume_calls, [*expected_calls, (second_start, None)])
        self.assertGreater(healthy_guard.call_count, 0)
        self.assertEqual(saved_path.read_bytes(), saved_bytes)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.observed_chunks, 2)
        self.assertEqual(result.failed_chunks, ())
        manifest = _read_json(self.out_dir / "_manifest.json")
        self.assertIs(manifest["production_ready"], False)
        self.assertEqual(manifest["total_rows"], 2)
        _, summary = audit_alpaca_news_history(self.out_dir)
        self.assertTrue(summary["passed"], summary["errors"])
        self.assertEqual(summary["page_count"], 3 if next_token else 2)


def _news(provider_id: int, timestamp: str, *, ticker: str = "FISV") -> dict[str, Any]:
    return {
        "id": provider_id,
        "created_at": timestamp,
        "updated_at": timestamp,
        "headline": f"SYNTHETIC TEST-ONLY issuer news {provider_id}",
        "source": "benzinga",
        "symbols": [ticker],
        "url": f"https://synthetic.example.test/news/{provider_id}",
        "summary": "SYNTHETIC TEST ONLY, not an observed provider article.",
        "content": "SYNTHETIC TEST ONLY, not production evidence.",
    }


def _empty_page(symbol: str, start: datetime, end: datetime, token: str | None) -> AlpacaNewsPage:
    return AlpacaNewsPage(token, None, ())


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
