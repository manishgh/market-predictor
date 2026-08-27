from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from market_predictor.canonical.audits import (
    CanonicalAuditReport,
    audit_universe_memberships,
)
from market_predictor.canonical.normalize import canonicalize_universe_memberships
from market_predictor.canonical.store import (
    load_canonical_artifact,
    write_canonical_artifact,
)
from market_predictor.catalysts.issuer_events.alpaca_news_audit import (
    audit_alpaca_news_history,
)
from market_predictor.catalysts.issuer_events.alpaca_news_collection import (
    collect_alpaca_news_history,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.sources.alpaca import AlpacaNewsPage


class SwingNewsHistoryTests(unittest.TestCase):
    def test_failed_page_resumes_without_refetch_and_publishes_research_events(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memberships = _memberships(root / "memberships.parquet")
            out_dir = root / "news"
            first_calls: list[str | None] = []

            def first_fetch(
                symbol: str,
                start: object,
                end: object,
                token: str | None,
            ) -> AlpacaNewsPage:
                del start, end
                self.assertEqual(symbol, "AAA")
                first_calls.append(token)
                if token == "page-2":
                    raise RuntimeError("temporary provider failure")
                return AlpacaNewsPage(
                    request_page_token=None,
                    next_page_token="page-2",
                    news=(_news(1, headline="Initial revision"),),
                )

            first = collect_alpaca_news_history(
                memberships_path=memberships,
                start_date=date(2026, 1, 1),
                end_date=date(2026, 1, 3),
                out_dir=out_dir,
                fetch_page=first_fetch,
                provider_symbol_for=lambda ticker: ticker,
                workers=1,
            )

            self.assertEqual(first.status, "incomplete")
            self.assertEqual(first_calls, [None, "page-2"])
            self.assertEqual(
                len(list((out_dir / "raw_pages").rglob("page_000000.json"))),
                1,
            )
            second_calls: list[str | None] = []

            def second_fetch(
                symbol: str,
                start: object,
                end: object,
                token: str | None,
            ) -> AlpacaNewsPage:
                del symbol, start, end
                second_calls.append(token)
                return AlpacaNewsPage(
                    request_page_token="page-2",
                    next_page_token=None,
                    news=(
                        _news(
                            1,
                            headline="Corrected revision",
                            updated_at="2026-01-02T11:00:00Z",
                        ),
                        _news(2, headline="Wrong symbol", symbols=["BBB"]),
                    ),
                )

            second = collect_alpaca_news_history(
                memberships_path=memberships,
                start_date=date(2026, 1, 1),
                end_date=date(2026, 1, 3),
                out_dir=out_dir,
                fetch_page=second_fetch,
                provider_symbol_for=lambda ticker: ticker,
                workers=1,
            )

            self.assertEqual(second.status, "complete")
            self.assertEqual(second_calls, ["page-2"])
            manifest = json.loads(
                (out_dir / "_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["production_ready"])
            self.assertEqual(manifest["total_rows"], 1)
            event_path = Path(manifest["artifacts"][0]["path"])
            events, event_manifest = load_canonical_artifact(
                event_path,
                expected_type="events",
                allow_research=True,
            )
            self.assertFalse(event_manifest["production_ready"])
            self.assertEqual(events.loc[0, "security_id"], "security:aaa")
            self.assertEqual(events.loc[0, "title"], "Corrected revision")
            self.assertEqual(
                events.loc[0, "published_at_utc"],
                pd.Timestamp("2026-01-02T10:00:00Z"),
            )
            self.assertEqual(
                events.loc[0, "available_at_utc"],
                pd.Timestamp("2026-01-02T11:00:00Z"),
            )
            self.assertEqual(
                events.loc[0, "availability_policy"],
                "provider_publication_proxy",
            )
            ledger, _ = load_canonical_artifact(
                out_dir / "_source_collections.parquet",
                expected_type="source_collections",
                allow_research=True,
            )
            self.assertEqual(int(ledger.loc[0, "provider_rows"]), 3)
            self.assertEqual(int(ledger.loc[0, "symbol_mismatch_rows"]), 1)
            self.assertEqual(int(ledger.loc[0, "duplicate_rows"]), 1)
            audit_report, audit_summary = audit_alpaca_news_history(out_dir)
            self.assertTrue(audit_summary["passed"])
            self.assertEqual(audit_summary["event_rows"], 1)
            self.assertEqual(audit_summary["page_count"], 2)
            self.assertEqual(
                audit_summary["coverage_blindspot_security_ids"],
                [],
            )
            self.assertEqual(audit_report.loc[0, "audit_errors"], "")
            self.assertTrue(audit_report.loc[0, "catalyst_source_complete"])

            with self.assertRaisesRegex(DataReadinessError, "immutable"):
                collect_alpaca_news_history(
                    memberships_path=memberships,
                    start_date=date(2026, 1, 1),
                    end_date=date(2026, 1, 3),
                    out_dir=out_dir,
                    fetch_page=second_fetch,
                    provider_symbol_for=lambda ticker: ticker,
                    workers=1,
                )

    def test_repeated_page_token_fails_without_final_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memberships = _memberships(root / "memberships.parquet")

            def fetch(
                symbol: str,
                start: object,
                end: object,
                token: str | None,
            ) -> AlpacaNewsPage:
                del symbol, start, end
                return AlpacaNewsPage(
                    request_page_token=token,
                    next_page_token="repeat",
                    news=(),
                )

            result = collect_alpaca_news_history(
                memberships_path=memberships,
                start_date=date(2026, 1, 1),
                end_date=date(2026, 1, 3),
                out_dir=root / "news",
                fetch_page=fetch,
                provider_symbol_for=lambda ticker: ticker,
                workers=1,
            )

            self.assertEqual(result.status, "incomplete")
            self.assertEqual(len(result.failed_chunks), 1)
            self.assertFalse((root / "news" / "_manifest.json").exists())

    def test_resume_rejects_modified_archived_page(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memberships = _memberships(root / "memberships.parquet")
            out_dir = root / "news"

            def initial_fetch(
                symbol: str,
                start: object,
                end: object,
                token: str | None,
            ) -> AlpacaNewsPage:
                del symbol, start, end
                if token is not None:
                    raise RuntimeError("stop after archived page")
                return AlpacaNewsPage(
                    request_page_token=None,
                    next_page_token="page-2",
                    news=(_news(1, headline="Original"),),
                )

            first = collect_alpaca_news_history(
                memberships_path=memberships,
                start_date=date(2026, 1, 1),
                end_date=date(2026, 1, 3),
                out_dir=out_dir,
                fetch_page=initial_fetch,
                provider_symbol_for=lambda ticker: ticker,
                workers=1,
            )
            self.assertEqual(first.status, "incomplete")
            page_path = next((out_dir / "raw_pages").rglob("page_000000.json"))
            payload = json.loads(page_path.read_text(encoding="utf-8"))
            payload["news"][0]["headline"] = "Modified"
            page_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            calls = 0

            def should_not_fetch(
                symbol: str,
                start: object,
                end: object,
                token: str | None,
            ) -> AlpacaNewsPage:
                nonlocal calls
                del symbol, start, end, token
                calls += 1
                return AlpacaNewsPage(None, None, ())

            second = collect_alpaca_news_history(
                memberships_path=memberships,
                start_date=date(2026, 1, 1),
                end_date=date(2026, 1, 3),
                out_dir=out_dir,
                fetch_page=should_not_fetch,
                provider_symbol_for=lambda ticker: ticker,
                workers=1,
            )
            self.assertEqual(second.status, "incomplete")
            self.assertEqual(calls, 0)


def _memberships(path: Path) -> Path:
    raw = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "security_id": ["security:aaa"],
            "effective_from_utc": [pd.Timestamp("2025-01-01T00:00:00Z")],
            "effective_to_utc": [pd.NaT],
            "available_at_utc": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "sector": ["Technology"],
            "industry": ["Software"],
            "market_cap_bucket": ["large"],
            "liquidity_bucket": ["high"],
            "primary_benchmark": ["XLK"],
            "universe_snapshot_id": ["test-memberships"],
            "source": ["test"],
            "availability_policy": ["provider_publication_proxy"],
        }
    )
    memberships = canonicalize_universe_memberships(raw)
    audit = CanonicalAuditReport(
        checks=audit_universe_memberships(
            memberships,
            require_observed=False,
        )
    )
    write_canonical_artifact(
        memberships,
        path,
        artifact_type="memberships",
        audit=audit,
        production_ready=False,
    )
    return path


def _news(
    provider_id: int,
    *,
    headline: str,
    updated_at: str = "2026-01-02T10:05:00Z",
    symbols: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": provider_id,
        "created_at": "2026-01-02T10:00:00Z",
        "updated_at": updated_at,
        "headline": headline,
        "source": "benzinga",
        "symbols": symbols or ["AAA"],
        "url": f"https://example.test/{provider_id}",
        "summary": "Summary",
        "content": "Content",
    }


if __name__ == "__main__":
    unittest.main()
