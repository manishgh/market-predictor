from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_predictor.v3.catalysts import (
    O1AuditConfig,
    O1OverlayConfig,
    build_o1_overlay_evidence,
    evaluate_o1_ablation,
)


class V3CatalystOverlayTests(unittest.TestCase):
    def test_o1_joins_only_available_events_and_deduplicates_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            _events("AAA", sentiment=0.9, available_at="2026-01-05T13:30:00Z").to_parquet(
                first / "AAA_events.parquet", index=False
            )
            _events("AAA", sentiment=0.9, available_at="2026-01-05T13:30:00Z").to_parquet(
                second / "AAA_events.parquet", index=False
            )
            _events("BBB", sentiment=-0.9, available_at="2026-01-05T13:35:00Z", offering=True).to_parquet(
                first / "BBB_events.parquet", index=False
            )
            _events("CCC", sentiment=0.2, available_at="2026-01-05T14:30:00Z").to_parquet(
                first / "CCC_events.parquet", index=False
            )
            evidence, audit = build_o1_overlay_evidence(
                _predictions(),
                event_directories=[first, second],
                market_context_path=None,
                config=_overlay_config(),
            )
        first_group = evidence[
            evidence["decision_time_utc"].eq(pd.Timestamp("2026-01-05T14:00:00Z"))
            & evidence["audit_scope"].eq("walk_forward")
        ].set_index("ticker")
        self.assertEqual(float(first_group.loc["AAA", "ticker_event_count_2h"]), 1.0)
        self.assertEqual(float(first_group.loc["CCC", "ticker_event_count_2h"]), 0.0)
        self.assertTrue(bool(first_group.loc["BBB", "o1_ticker_veto"]))
        self.assertGreater(float(first_group.loc["AAA", "o1_score"]), float(first_group.loc["BBB", "o1_score"]))
        self.assertEqual(audit["future_matches"], 0)
        self.assertTrue(audit["ready"])

    def test_publication_backfill_is_explicitly_research_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_dir = Path(directory)
            for ticker in ("AAA", "BBB", "CCC"):
                events = _events(ticker, sentiment=0.6, available_at="2026-01-05T13:30:00Z").drop(
                    columns="available_at_utc"
                )
                events.to_parquet(event_dir / f"{ticker}_events.parquet", index=False)
            evidence, audit = build_o1_overlay_evidence(
                _predictions(),
                event_directories=[event_dir],
                market_context_path=None,
                config=_overlay_config(availability_policy="provider_publication_backfill"),
            )
        self.assertFalse(evidence.empty)
        self.assertTrue(audit["ready"])
        self.assertTrue(audit["research_only"])

    def test_missing_sentiment_fails_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_dir = Path(directory)
            for ticker in ("AAA", "BBB", "CCC"):
                _events(ticker, sentiment=0.5, available_at="2026-01-05T13:30:00Z").drop(
                    columns="sentiment_numeric"
                ).to_parquet(event_dir / f"{ticker}_events.parquet", index=False)
            _, audit = build_o1_overlay_evidence(
                _predictions(),
                event_directories=[event_dir],
                market_context_path=None,
                config=_overlay_config(),
            )
        self.assertFalse(audit["ready"])
        self.assertIn("sentiment coverage", " ".join(audit["readiness_failures"]))

    def test_stale_market_context_fails_declared_coverage_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            event_dir = root / "events"
            event_dir.mkdir()
            for ticker in ("AAA", "BBB", "CCC"):
                _events(ticker, sentiment=0.5, available_at="2026-01-05T13:30:00Z").to_parquet(
                    event_dir / f"{ticker}_events.parquet", index=False
                )
            market_context = root / "market.parquet"
            _events("MARKET", sentiment=-0.5, available_at="2026-01-02T12:00:00Z").to_parquet(
                market_context, index=False
            )
            _, audit = build_o1_overlay_evidence(
                _predictions(),
                event_directories=[event_dir],
                market_context_path=market_context,
                config=_overlay_config(),
            )

        self.assertFalse(audit["ready"])
        self.assertIn("end boundary", " ".join(audit["readiness_failures"]))

    def test_o1_ablation_uses_paired_scopes_and_improves_synthetic_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_dir = Path(directory)
            _events("AAA", sentiment=0.95, available_at="2026-01-05T13:30:00Z").to_parquet(
                event_dir / "AAA_events.parquet", index=False
            )
            _events("BBB", sentiment=-0.95, available_at="2026-01-05T13:30:00Z", offering=True).to_parquet(
                event_dir / "BBB_events.parquet", index=False
            )
            _events("CCC", sentiment=0.0, available_at="2026-01-05T13:30:00Z").to_parquet(
                event_dir / "CCC_events.parquet", index=False
            )
            evidence, _ = build_o1_overlay_evidence(
                _predictions(),
                event_directories=[event_dir],
                market_context_path=None,
                config=_overlay_config(overlay_weight=0.5),
            )
            report, selected = evaluate_o1_ablation(
                evidence,
                config=O1AuditConfig(top_k=1, bootstrap_iterations=100, minimum_sessions=2),
            )
        self.assertEqual(set(report["scopes"]), {"walk_forward", "ticker_holdout"})
        self.assertGreater(
            report["scopes"]["walk_forward"]["o1"]["mean_top_k_excess_return"],
            report["scopes"]["walk_forward"]["r1"]["mean_top_k_excess_return"],
        )
        self.assertEqual(set(selected["strategy"]), {"R1", "O1"})


def _overlay_config(**updates: object) -> O1OverlayConfig:
    values: dict[str, object] = {
        "coverage_start_utc": pd.Timestamp("2026-01-01T00:00:00Z").to_pydatetime(),
        "coverage_end_utc": pd.Timestamp("2026-01-07T23:59:59Z").to_pydatetime(),
        "minimum_ticker_file_coverage": 1.0,
        "minimum_sentiment_coverage": 1.0,
        "minimum_decision_rows": 1,
    }
    values.update(updates)
    return O1OverlayConfig(**values)  # type: ignore[arg-type]


def _events(
    ticker: str,
    *,
    sentiment: float,
    available_at: str,
    offering: bool = False,
) -> pd.DataFrame:
    title = f"{ticker} announces {'stock offering' if offering else 'major contract win'}"
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "timestamp": "2026-01-05T13:25:00Z",
                "available_at_utc": available_at,
                "source": "alpaca:benzinga",
                "title": title,
                "url": f"https://example.test/{ticker}",
                "summary": title,
                "text": title,
                "sentiment_numeric": sentiment,
            }
        ]
    )


def _predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in ("walk_forward", "ticker_holdout"):
        for session_index, day in enumerate(("2026-01-05", "2026-01-06")):
            decision = pd.Timestamp(f"{day}T14:00:00Z")
            for ticker, score, target, grade in (
                ("AAA", 0.4, 0.03, 2),
                ("BBB", 0.9, -0.02, 0),
                ("CCC", 0.2, 0.00, 1),
            ):
                rows.append(
                    {
                        "ticker": ticker,
                        "decision_time_utc": decision,
                        "decision_group_id": f"{scope}-{day}",
                        "audit_scope": scope,
                        "family": "R1",
                        "model_run_id": "r1-run",
                        "score": score,
                        "ranking_target": target,
                        "ranking_grade": grade,
                        "session_date_et": day,
                        "entry_time_utc": decision + pd.Timedelta(minutes=5),
                        "primary_exit_time_utc": decision + pd.Timedelta(minutes=30),
                        "path_realized_return_net": target,
                        "independent_event_id": f"{scope}-{session_index}-{ticker}",
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
