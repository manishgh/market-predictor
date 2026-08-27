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
from market_predictor.catalysts.issuer_events.alpaca_news_audit import (
    audit_alpaca_news_history,
)
from market_predictor.catalysts.issuer_events.alpaca_news_collection import (
    collect_alpaca_news_history,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.sources.alpaca import AlpacaNewsPage
from market_predictor.swing.sentiment_history import (
    SENTIMENT_AVAILABILITY_POLICY,
    score_alpaca_news_history,
)


class _DeterministicScorer:
    def __init__(
        self,
        *,
        fail_call: int | None = None,
        fail_calls: set[int] | None = None,
    ) -> None:
        self.calls = 0
        self.call_sizes: list[int] = []
        self.fail_calls = set(fail_calls or set())
        if fail_call is not None:
            self.fail_calls.add(fail_call)

    def score_texts(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> pd.DataFrame:
        del batch_size
        self.calls += 1
        self.call_sizes.append(len(texts))
        if self.calls in self.fail_calls:
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
                max_batch_shards=1,
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
                max_batch_shards=8,
            )

            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resumed["observed_chunks"], 2)
            self.assertEqual(resumed["skipped_chunks"], 1)
            self.assertEqual(resumed_scorer.calls, 1)
            self.assertEqual(resumed["scorer_calls"], 1)
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

    def test_cross_shard_batch_scores_once_and_preserves_chunk_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection_dir, audit_path, universe_path = _archive(root)
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
                max_batch_events=10,
                max_batch_shards=10,
            )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["scorer_calls"], 1)
            self.assertEqual(scorer.calls, 1)
            self.assertEqual(
                [artifact["ticker"] for artifact in result["artifacts"]],
                ["AAA", "BBB"],
            )
            for artifact in result["artifacts"]:
                frame, manifest = load_canonical_artifact(
                    Path(artifact["path"]),
                    expected_type="event_sentiment_research",
                    allow_research=True,
                )
                self.assertEqual(len(frame), 1)
                self.assertEqual(frame.loc[0, "ticker"], artifact["ticker"])
                self.assertEqual(manifest["artifact_sha256"], artifact["sha256"])

    def test_batch_failure_records_each_chunk_and_publishes_none(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection_dir, audit_path, universe_path = _archive(root)

            result = score_alpaca_news_history(
                collection_dir=collection_dir,
                collection_audit_path=audit_path,
                universe_path=universe_path,
                out_dir=root / "sentiment",
                scorer=_DeterministicScorer(fail_call=1),
                model_name="test-finbert",
                model_revision="revision-1",
                execution_device="cpu",
                max_batch_events=10,
                max_batch_shards=10,
            )

            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["observed_chunks"], 0)
            self.assertEqual(len(result["failed_chunks"]), 2)
            self.assertEqual(result["scorer_calls"], 1)
            self.assertEqual(
                len(list((root / "sentiment" / "attempts").glob("*.json"))),
                2,
            )
            self.assertEqual(
                list((root / "sentiment" / "sentiment").glob("*.parquet")),
                [],
            )

    def test_event_bound_splits_batches_without_changing_chunk_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection_dir, audit_path, universe_path = _archive(root)
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
                max_batch_events=1,
                max_batch_shards=10,
            )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["scorer_calls"], 2)
            self.assertEqual(scorer.calls, 2)
            self.assertEqual(
                [artifact["ticker"] for artifact in result["artifacts"]],
                ["AAA", "BBB"],
            )

    def test_empty_source_shards_publish_deterministic_zero_row_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection_dir, audit_path, universe_path = _archive(
                root,
                events_per_ticker={"AAA": 0, "BBB": 0},
            )
            artifact_hashes: list[list[str]] = []
            for suffix in ("first", "second"):
                scorer = _DeterministicScorer()
                result = score_alpaca_news_history(
                    collection_dir=collection_dir,
                    collection_audit_path=audit_path,
                    universe_path=universe_path,
                    out_dir=root / f"sentiment-{suffix}",
                    scorer=scorer,
                    model_name="test-finbert",
                    model_revision="revision-1",
                    execution_device="cpu",
                )

                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["requested_chunks"], 2)
                self.assertEqual(result["observed_chunks"], 2)
                self.assertEqual(result["total_rows"], 0)
                self.assertEqual(result["scorer_calls"], 0)
                self.assertEqual(scorer.calls, 0)
                artifact_hashes.append(
                    [artifact["sha256"] for artifact in result["artifacts"]]
                )
                for artifact in result["artifacts"]:
                    frame, _ = load_canonical_artifact(
                        Path(artifact["path"]),
                        expected_type="event_sentiment_research",
                        allow_research=True,
                    )
                    self.assertTrue(frame.empty)
                    self.assertIsNone(
                        artifact["first_feature_available_at_utc"]
                    )
                    self.assertIsNone(
                        artifact["last_feature_available_at_utc"]
                    )
            self.assertEqual(artifact_hashes[0], artifact_hashes[1])

    def test_large_source_shard_is_scored_in_slices_and_published_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection_dir, audit_path, universe_path = _archive(
                root,
                tickers=("AAA",),
                events_per_ticker={"AAA": 5},
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
                max_batch_events=2,
                max_batch_shards=10,
            )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["scorer_calls"], 3)
            self.assertEqual(scorer.call_sizes, [2, 2, 1])
            self.assertEqual(len(result["artifacts"]), 1)
            self.assertEqual(result["artifacts"][0]["rows"], 5)
            frame, _ = load_canonical_artifact(
                Path(result["artifacts"][0]["path"]),
                expected_type="event_sentiment_research",
                allow_research=True,
            )
            self.assertEqual(len(frame), 5)
            self.assertEqual(frame["event_id"].nunique(), 5)

    def test_resume_skips_middle_chunk_without_flushing_pending_batch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection_dir, audit_path, universe_path = _archive(
                root,
                tickers=("AAA", "BBB", "CCC"),
            )
            out_dir = root / "sentiment"
            first = score_alpaca_news_history(
                collection_dir=collection_dir,
                collection_audit_path=audit_path,
                universe_path=universe_path,
                out_dir=out_dir,
                scorer=_DeterministicScorer(fail_calls={1, 3}),
                model_name="test-finbert",
                model_revision="revision-1",
                execution_device="cpu",
                max_batch_shards=1,
            )
            self.assertEqual(first["status"], "incomplete")
            self.assertEqual(first["observed_chunks"], 1)

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
                max_batch_shards=10,
            )

            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resumed_scorer.calls, 1)
            self.assertEqual(resumed_scorer.call_sizes, [2])
            self.assertEqual(
                [artifact["ticker"] for artifact in resumed["artifacts"]],
                ["AAA", "BBB", "CCC"],
            )


def _archive(
    root: Path,
    *,
    tickers: tuple[str, ...] = ("AAA", "BBB"),
    events_per_ticker: dict[str, int] | None = None,
) -> tuple[Path, Path, Path]:
    memberships_path = _memberships(
        root / "memberships.parquet",
        tickers=tickers,
    )
    collection_dir = root / "news"
    event_counts = events_per_ticker or {ticker: 1 for ticker in tickers}
    ticker_ids = {ticker: index + 1 for index, ticker in enumerate(tickers)}

    def fetch(
        symbol: str,
        start: object,
        end: object,
        token: str | None,
    ) -> AlpacaNewsPage:
        del start, end
        if token is not None:
            raise AssertionError("fixture has one page")
        news = tuple(
            {
                "id": ticker_ids[symbol] * 1_000 + index,
                "created_at": "2026-01-02T10:00:00Z",
                "updated_at": "2026-01-02T10:05:00Z",
                "headline": f"{symbol} wins contract {index}",
                "source": "benzinga",
                "symbols": [symbol],
                "url": f"https://example.test/{symbol}/{index}",
                "summary": "Revenue guidance increased.",
                "content": "Management discussed the contract.",
            }
            for index in range(event_counts.get(symbol, 0))
        )
        return AlpacaNewsPage(
            request_page_token=None,
            next_page_token=None,
            news=news,
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
            "ticker": list(tickers),
            "security_id": [f"security:{ticker.lower()}" for ticker in tickers],
            "company": [f"{ticker} Systems" for ticker in tickers],
            "sector": ["Information Technology"] * len(tickers),
            "industry": ["Software"] * len(tickers),
        }
    ).to_parquet(universe_path, index=False)
    return collection_dir, audit_path, universe_path


def _memberships(
    path: Path,
    *,
    tickers: tuple[str, ...] = ("AAA", "BBB"),
) -> Path:
    raw = pd.DataFrame(
        {
            "ticker": list(tickers),
            "security_id": [f"security:{ticker.lower()}" for ticker in tickers],
            "effective_from_utc": [
                pd.Timestamp("2025-01-01T00:00:00Z")
            ] * len(tickers),
            "effective_to_utc": [pd.NaT] * len(tickers),
            "available_at_utc": [
                pd.Timestamp("2026-01-01T00:00:00Z")
            ] * len(tickers),
            "sector": ["Information Technology"] * len(tickers),
            "industry": ["Software"] * len(tickers),
            "market_cap_bucket": ["large"] * len(tickers),
            "liquidity_bucket": ["high"] * len(tickers),
            "primary_benchmark": ["XLK"] * len(tickers),
            "universe_snapshot_id": ["test-memberships"] * len(tickers),
            "source": ["test"] * len(tickers),
            "availability_policy": ["provider_publication_proxy"]
            * len(tickers),
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
