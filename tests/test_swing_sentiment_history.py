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
from market_predictor.canonical.normalize import (
    canonicalize_universe_memberships,
)
from market_predictor.canonical.store import (
    load_canonical_artifact,
    write_canonical_artifact,
)
from market_predictor.sources.alpaca import AlpacaNewsPage
from market_predictor.swing.news_history import collect_alpaca_news_history
from market_predictor.swing.news_history_audit import (
    audit_alpaca_news_history,
)
from market_predictor.swing.sentiment_history import (
    SENTIMENT_AVAILABILITY_POLICY,
    score_alpaca_news_history,
)
from market_predictor.v3.errors import DataReadinessError


class _DeterministicScorer:
    def __init__(self, *, fail_call: int | None = None) -> None:
        self.calls = 0
        self.fail_call = fail_call

    def score_texts(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> pd.DataFrame:
        del batch_size
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError("injected scorer failure")
        return pd.DataFrame(
            {
                "sentiment_label": ["positive"] * len(texts),
                "sentiment_score": [0.8] * len(texts),
                "sentiment_numeric": [0.8] * len(texts),
            }
        )


class SwingSentimentHistoryTests(unittest.TestCase):
    def test_partial_failure_resumes_without_rescoring_completed_chunk(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection_dir, audit_path, universe_path = _archive(root)
            out_dir = root / "sentiment"
            first_scorer = _DeterministicScorer(fail_call=2)

            first = score_alpaca_news_history(
                collection_dir=collection_dir,
                collection_audit_path=audit_path,
                universe_path=universe_path,
                out_dir=out_dir,
                scorer=first_scorer,
                model_name="test-finbert",
                model_revision="revision-1",
                execution_device="cpu",
                fixed_latency_minutes=5,
            )

            self.assertEqual(first["status"], "incomplete")
            self.assertEqual(first["observed_chunks"], 1)
            self.assertEqual(len(first["failed_chunks"]), 1)
            self.assertEqual(first_scorer.calls, 2)

            resumed_scorer = _DeterministicScorer()
            resumed = score_alpaca_news_history(
                collection_dir=collection_dir,
                collection_audit_path=audit_path,
                universe_path=universe_path,
                out_dir=out_dir,
                scorer=resumed_scorer,
                model_name="test-finbert",
                model_revision="revision-1",
                execution_device="cpu",
                fixed_latency_minutes=5,
            )

            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resumed["observed_chunks"], 2)
            self.assertEqual(resumed["skipped_chunks"], 1)
            self.assertEqual(resumed_scorer.calls, 1)
            self.assertEqual(
                {item["ticker"] for item in resumed["artifacts"]},
                {"AAA", "BBB"},
            )
            frame, manifest = load_canonical_artifact(
                Path(resumed["artifacts"][0]["path"]),
                expected_type="event_sentiment_research",
                allow_research=True,
            )
            event_available = pd.to_datetime(
                frame["event_available_at_utc"],
                utc=True,
            )
            feature_available = pd.to_datetime(
                frame["research_feature_available_at_utc"],
                utc=True,
            )
            computed = pd.to_datetime(
                frame["inference_computed_at_utc"],
                utc=True,
            )
            self.assertTrue(
                feature_available.eq(
                    event_available + pd.Timedelta(minutes=5)
                ).all()
            )
            self.assertTrue(computed.ge(event_available).all())
            self.assertEqual(
                frame.loc[0, "sentiment_availability_policy"],
                SENTIMENT_AVAILABILITY_POLICY,
            )
            self.assertEqual(frame.loc[0, "sentiment_model_revision"], "revision-1")
            self.assertGreater(float(frame.loc[0, "relevance"]), 0.0)
            self.assertFalse(manifest["production_ready"])
            self.assertEqual(len(frame.loc[0, "sentiment_input_sha256"]), 64)

            with self.assertRaisesRegex(DataReadinessError, "immutable"):
                score_alpaca_news_history(
                    collection_dir=collection_dir,
                    collection_audit_path=audit_path,
                    universe_path=universe_path,
                    out_dir=out_dir,
                    scorer=_DeterministicScorer(),
                    model_name="test-finbert",
                    model_revision="revision-1",
                    execution_device="cpu",
                )

    def test_audit_coverage_blindspot_is_excluded_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection_dir, audit_path, universe_path = _archive(root)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["coverage_blindspot_security_ids"] = ["security:bbb"]
            audit_path.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            scorer = _DeterministicScorer()

            result = score_alpaca_news_history(
                collection_dir=collection_dir,
                collection_audit_path=audit_path,
                universe_path=universe_path,
                out_dir=root / "sentiment",
                scorer=scorer,
                model_name="test-finbert",
                model_revision="revision-1",
                execution_device="cpu",
            )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["requested_chunks"], 1)
            self.assertEqual(result["excluded_chunks"], 1)
            self.assertEqual(
                result["excluded_security_ids"],
                ["security:bbb"],
            )
            self.assertEqual(scorer.calls, 1)


def _archive(root: Path) -> tuple[Path, Path, Path]:
    memberships_path = _memberships(root / "memberships.parquet")
    collection_dir = root / "news"

    def fetch(
        symbol: str,
        start: object,
        end: object,
        token: str | None,
    ) -> AlpacaNewsPage:
        del start, end
        if token is not None:
            raise AssertionError("fixture has one page")
        provider_id = 1 if symbol == "AAA" else 2
        return AlpacaNewsPage(
            request_page_token=None,
            next_page_token=None,
            news=(
                {
                    "id": provider_id,
                    "created_at": "2026-01-02T10:00:00Z",
                    "updated_at": "2026-01-02T10:05:00Z",
                    "headline": f"{symbol} wins a material software contract",
                    "source": "benzinga",
                    "symbols": [symbol],
                    "url": f"https://example.test/{provider_id}",
                    "summary": "Revenue guidance increased.",
                    "content": "Management discussed the contract.",
                },
            ),
        )

    result = collect_alpaca_news_history(
        memberships_path=memberships_path,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 3),
        out_dir=collection_dir,
        fetch_page=fetch,
        provider_symbol_for=lambda ticker: ticker,
        workers=1,
    )
    if result.status != "complete":
        raise AssertionError(result)
    _, summary = audit_alpaca_news_history(collection_dir)
    audit_path = root / "audit.json"
    audit_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    universe_path = root / "universe.parquet"
    pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "security_id": ["security:aaa", "security:bbb"],
            "company": ["Alpha Analytics", "Beta Systems"],
            "sector": ["Information Technology", "Information Technology"],
            "industry": ["Software", "Software"],
        }
    ).to_parquet(universe_path, index=False)
    return collection_dir, audit_path, universe_path


def _memberships(path: Path) -> Path:
    raw = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "security_id": ["security:aaa", "security:bbb"],
            "effective_from_utc": [
                pd.Timestamp("2025-01-01T00:00:00Z"),
                pd.Timestamp("2025-01-01T00:00:00Z"),
            ],
            "effective_to_utc": [pd.NaT, pd.NaT],
            "available_at_utc": [
                pd.Timestamp("2026-01-01T00:00:00Z"),
                pd.Timestamp("2026-01-01T00:00:00Z"),
            ],
            "sector": ["Information Technology", "Information Technology"],
            "industry": ["Software", "Software"],
            "market_cap_bucket": ["large", "large"],
            "liquidity_bucket": ["high", "high"],
            "primary_benchmark": ["XLK", "XLK"],
            "universe_snapshot_id": ["test-memberships", "test-memberships"],
            "source": ["test", "test"],
            "availability_policy": [
                "provider_publication_proxy",
                "provider_publication_proxy",
            ],
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


if __name__ == "__main__":
    unittest.main()
