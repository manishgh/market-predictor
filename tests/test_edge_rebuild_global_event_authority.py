from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import pandas as pd
import pytest

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.contracts import CanonicalEvent, SourceCollection
from market_predictor.canonical.store import write_canonical_artifact
from market_predictor.edge_rebuild.global_event_authority import (
    attach_global_event_features,
    load_global_event_authority,
    publish_global_event_authority,
)
from market_predictor.core.errors import DataReadinessError

DECISION_TIME = pd.Timestamp("2025-01-10T21:00:00Z")
LATER_DECISION_TIME = DECISION_TIME + pd.Timedelta(hours=2)
COLLECTION_REQUEST_SHA256 = "c" * 64
SOURCE_POLICY_SHA256 = "d" * 64
SCORER_IDENTITY = "fixture-finbert|revision=v1|max_length=128"


def test_production_authority_is_causal_explicit_and_exactly_attachable(
    tmp_path: Path,
) -> None:
    events = _events(
        [
            _event("1", "alpaca", DECISION_TIME - pd.Timedelta(hours=6), 0.6),
            _event("2", "alpaca", DECISION_TIME + pd.Timedelta(hours=1), -0.2),
        ]
    )
    coverage = _coverage(
        decisions=(DECISION_TIME, LATER_DECISION_TIME),
        sources=("alpaca", "gdelt"),
        alpaca_rows=(1, 2),
    )
    event_path = _write_events(tmp_path / "events.parquet", events, production=True)
    coverage_path = _write_coverage(
        tmp_path / "coverage.parquet",
        coverage,
        production=True,
    )

    authority = publish_global_event_authority(
        pd.DataFrame(
            {
                "decision_time_utc": [
                    DECISION_TIME,
                    DECISION_TIME,
                    LATER_DECISION_TIME,
                ]
            }
        ),
        [event_path],
        [coverage_path],
        tmp_path / "authority",
        required_historical_sources=("gdelt", "alpaca"),
        production_ready=True,
    )

    assert authority.manifest["required_historical_sources"] == ["alpaca", "gdelt"]
    assert authority.manifest["production_ready"] is True
    assert authority.decisions["global_event_count_1d"].tolist() == [1.0, 2.0]
    assert authority.decisions["global_source_count_alpaca_3d"].tolist() == [1.0, 2.0]
    assert authority.decisions["global_source_count_gdelt_3d"].tolist() == [0.0, 0.0]
    assert authority.decisions["global_source_complete_3d"].tolist() == [True, True]
    assert authority.decisions["global_sentiment_mean_1d"].tolist() == pytest.approx([0.6, 0.2])
    assert authority.decisions.loc[0, "global_latest_event_feature_available_at_utc_3d"] < DECISION_TIME

    attached = attach_global_event_features(
        pd.DataFrame(
            {
                "decision_id": ["ticker-a", "ticker-b"],
                "decision_time_utc": [DECISION_TIME, DECISION_TIME],
            }
        ),
        authority,
    )
    assert attached["global_event_count_3d"].tolist() == [1.0, 1.0]


def test_post_decision_collection_is_research_only_and_production_unknown(
    tmp_path: Path,
) -> None:
    event_path = _write_events(
        tmp_path / "events.parquet",
        _events([_event("1", "alpaca", DECISION_TIME - pd.Timedelta(hours=6), 0.4)]),
        production=True,
    )
    coverage = _coverage(
        decisions=(DECISION_TIME,),
        sources=("alpaca",),
        alpaca_rows=(1,),
        collection_lag=pd.Timedelta(days=30),
    )
    coverage_path = _write_coverage(
        tmp_path / "coverage.parquet",
        coverage,
        production=True,
    )

    authority = publish_global_event_authority(
        pd.DataFrame({"decision_time_utc": [DECISION_TIME]}),
        [event_path],
        [coverage_path],
        tmp_path / "research-authority",
        required_historical_sources=("alpaca",),
        production_ready=False,
    )

    assert bool(authority.decisions.loc[0, "global_source_complete_3d"])
    assert authority.decisions.loc[0, "global_event_count_3d"] == 1.0
    request = cast(dict[str, object], authority.manifest["request"])
    assert request["coverage_completion_policy"] == (
        "retrospective research backfill may complete after decision_time_utc"
    )

    with pytest.raises(DataReadinessError, match="complete explicit source coverage"):
        publish_global_event_authority(
            pd.DataFrame({"decision_time_utc": [DECISION_TIME]}),
            [event_path],
            [coverage_path],
            tmp_path / "production-authority",
            required_historical_sources=("alpaca",),
            production_ready=True,
        )


def test_sentiment_without_relevance_uses_equal_weight(tmp_path: Path) -> None:
    event = _event(
        "1",
        "alpaca",
        DECISION_TIME - pd.Timedelta(hours=6),
        0.4,
    )
    event["relevance"] = None
    event_path = _write_events(
        tmp_path / "events.parquet",
        _events([event]),
        production=True,
    )
    coverage_path = _write_coverage(
        tmp_path / "coverage.parquet",
        _coverage(
            decisions=(DECISION_TIME,),
            sources=("alpaca",),
            alpaca_rows=(1,),
        ),
        production=True,
    )

    authority = publish_global_event_authority(
        pd.DataFrame({"decision_time_utc": [DECISION_TIME]}),
        [event_path],
        [coverage_path],
        tmp_path / "authority",
        required_historical_sources=("alpaca",),
        production_ready=True,
    )

    assert authority.decisions.loc[0, "global_sentiment_mean_1d"] == pytest.approx(0.4)


def test_research_authority_preserves_unknown_instead_of_fabricating_zero(
    tmp_path: Path,
) -> None:
    event_path = _write_events(
        tmp_path / "events.parquet",
        _events([_event("1", "alpaca", DECISION_TIME - pd.Timedelta(hours=6), 0.4)]),
        production=False,
    )
    coverage_path = _write_coverage(
        tmp_path / "coverage.parquet",
        _coverage(
            decisions=(DECISION_TIME,),
            sources=("alpaca",),
            alpaca_rows=(1,),
        ),
        production=False,
    )

    authority = publish_global_event_authority(
        pd.DataFrame({"decision_time_utc": [DECISION_TIME]}),
        [event_path],
        [coverage_path],
        tmp_path / "authority",
        required_historical_sources=("alpaca", "gdelt"),
        production_ready=False,
    )

    row = authority.decisions.iloc[0]
    assert bool(row["global_source_coverage_known_alpaca_3d"])
    assert row["global_source_count_alpaca_3d"] == 1.0
    assert not bool(row["global_source_coverage_known_gdelt_3d"])
    assert pd.isna(row["global_source_count_gdelt_3d"])
    assert pd.isna(row["global_event_count_3d"])
    with pytest.raises(DataReadinessError, match="not production ready"):
        load_global_event_authority(authority.directory)


def test_production_rejects_research_proxy_events(tmp_path: Path) -> None:
    proxy = _event(
        "1",
        "alpaca",
        DECISION_TIME - pd.Timedelta(hours=6),
        0.4,
        availability_policy="provider_publication_proxy",
    )
    event_path = _write_events(
        tmp_path / "events.parquet",
        _events([proxy]),
        production=False,
    )
    coverage_path = _write_coverage(
        tmp_path / "coverage.parquet",
        _coverage(
            decisions=(DECISION_TIME,),
            sources=("alpaca",),
            alpaca_rows=(1,),
        ),
        production=True,
    )

    with pytest.raises(DataReadinessError, match="research|production"):
        publish_global_event_authority(
            pd.DataFrame({"decision_time_utc": [DECISION_TIME]}),
            [event_path],
            [coverage_path],
            tmp_path / "authority",
            required_historical_sources=("alpaca",),
            production_ready=True,
        )


def test_rejects_ticker_events_and_undeclared_global_sources(tmp_path: Path) -> None:
    coverage_path = _write_coverage(
        tmp_path / "coverage.parquet",
        _coverage(
            decisions=(DECISION_TIME,),
            sources=("alpaca",),
            alpaca_rows=(1,),
        ),
        production=False,
    )
    ticker_event = _event(
        "1",
        "alpaca",
        DECISION_TIME - pd.Timedelta(hours=1),
        0.2,
        ticker="AAPL",
        security_id="security:aapl",
    )
    ticker_path = _write_events(
        tmp_path / "ticker-events.parquet",
        _events([ticker_event]),
        production=False,
    )
    with pytest.raises(DataReadinessError, match="rejects ticker events"):
        publish_global_event_authority(
            pd.DataFrame({"decision_time_utc": [DECISION_TIME]}),
            [ticker_path],
            [coverage_path],
            tmp_path / "ticker-authority",
            required_historical_sources=("alpaca",),
            production_ready=False,
        )

    undeclared_path = _write_events(
        tmp_path / "undeclared-events.parquet",
        _events(
            [
                _event(
                    "2",
                    "finviz",
                    DECISION_TIME - pd.Timedelta(hours=1),
                    0.2,
                )
            ]
        ),
        production=False,
    )
    with pytest.raises(DataReadinessError, match="not explicitly declared"):
        publish_global_event_authority(
            pd.DataFrame({"decision_time_utc": [DECISION_TIME]}),
            [undeclared_path],
            [coverage_path],
            tmp_path / "undeclared-authority",
            required_historical_sources=("alpaca",),
            production_ready=False,
        )


def test_loader_rejects_tampering_and_attachment_rejects_asof_fallback(
    tmp_path: Path,
) -> None:
    event_path = _write_events(
        tmp_path / "events.parquet",
        _events([_event("1", "alpaca", DECISION_TIME - pd.Timedelta(hours=1), 0.2)]),
        production=True,
    )
    coverage_path = _write_coverage(
        tmp_path / "coverage.parquet",
        _coverage(
            decisions=(DECISION_TIME,),
            sources=("alpaca",),
            alpaca_rows=(1,),
        ),
        production=True,
    )
    authority = publish_global_event_authority(
        pd.DataFrame({"decision_time_utc": [DECISION_TIME]}),
        [event_path],
        [coverage_path],
        tmp_path / "authority",
        required_historical_sources=("alpaca",),
        production_ready=True,
    )
    with pytest.raises(DataReadinessError, match="no exact row"):
        attach_global_event_features(
            pd.DataFrame({"decision_time_utc": [DECISION_TIME + pd.Timedelta(seconds=1)]}),
            authority,
        )

    artifact = authority.directory / "decision_global_events.parquet"
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(DataReadinessError, match="integrity"):
        load_global_event_authority(authority.directory)


def test_rejects_unmatched_collection_lineage_and_memory_budget_above_four_gib(
    tmp_path: Path,
) -> None:
    event_path = _write_events(
        tmp_path / "events.parquet",
        _events([_event("1", "alpaca", DECISION_TIME - pd.Timedelta(hours=1), 0.2)]),
        production=False,
        collection_request_sha256="a" * 64,
    )
    coverage_path = _write_coverage(
        tmp_path / "coverage.parquet",
        _coverage(
            decisions=(DECISION_TIME,),
            sources=("alpaca",),
            alpaca_rows=(1,),
        ),
        production=False,
        collection_request_sha256="b" * 64,
    )
    decisions = pd.DataFrame({"decision_time_utc": [DECISION_TIME]})

    with pytest.raises(DataReadinessError, match="no matching source-coverage lineage"):
        publish_global_event_authority(
            decisions,
            [event_path],
            [coverage_path],
            tmp_path / "lineage-authority",
            required_historical_sources=("alpaca",),
            production_ready=False,
        )
    with pytest.raises(DataReadinessError, match="cannot exceed 4 GiB"):
        publish_global_event_authority(
            decisions,
            [event_path],
            [coverage_path],
            tmp_path / "memory-authority",
            required_historical_sources=("alpaca",),
            production_ready=False,
            maximum_process_memory_gib=4.1,
        )


def test_rejects_mixed_source_policies_for_one_global_family(tmp_path: Path) -> None:
    event_path = _write_events(
        tmp_path / "events.parquet",
        _events([_event("1", "alpaca", DECISION_TIME - pd.Timedelta(hours=1), 0.2)]),
        production=False,
        source_policy_sha256="1" * 64,
    )
    coverage_path = _write_coverage(
        tmp_path / "coverage.parquet",
        _coverage(
            decisions=(DECISION_TIME,),
            sources=("alpaca",),
            alpaca_rows=(1,),
        ),
        production=False,
        source_policy_sha256="2" * 64,
    )

    with pytest.raises(DataReadinessError, match="source policy is inconsistent"):
        publish_global_event_authority(
            pd.DataFrame({"decision_time_utc": [DECISION_TIME]}),
            [event_path],
            [coverage_path],
            tmp_path / "authority",
            required_historical_sources=("alpaca",),
            production_ready=False,
        )


@pytest.mark.parametrize(
    ("status", "row_count", "message"),
    [
        ("observed_empty", 0, "row_count does not reconcile"),
        ("observed", 2, "row_count does not reconcile"),
    ],
)
def test_rejects_false_zero_and_tampered_collection_counts(
    tmp_path: Path,
    status: str,
    row_count: int,
    message: str,
) -> None:
    event_path = _write_events(
        tmp_path / "events.parquet",
        _events(
            [_event("1", "alpaca", DECISION_TIME - pd.Timedelta(hours=1), 0.2)]
        ),
        production=False,
    )
    coverage = _coverage(
        decisions=(DECISION_TIME,),
        sources=("alpaca",),
        alpaca_rows=(1,),
    )
    coverage.loc[0, "status"] = status
    coverage.loc[0, "row_count"] = row_count
    coverage_path = _write_coverage(
        tmp_path / "coverage.parquet",
        coverage,
        production=False,
    )

    with pytest.raises(DataReadinessError, match=message):
        publish_global_event_authority(
            pd.DataFrame({"decision_time_utc": [DECISION_TIME]}),
            [event_path],
            [coverage_path],
            tmp_path / "authority",
            required_historical_sources=("alpaca",),
            production_ready=False,
        )


def test_rejects_event_outside_matching_collection_window(tmp_path: Path) -> None:
    event_path = _write_events(
        tmp_path / "events.parquet",
        _events(
            [_event("1", "alpaca", DECISION_TIME - pd.Timedelta(days=4), 0.2)]
        ),
        production=False,
    )
    coverage_path = _write_coverage(
        tmp_path / "coverage.parquet",
        _coverage(
            decisions=(DECISION_TIME,),
            sources=("alpaca",),
            alpaca_rows=(0,),
        ),
        production=False,
    )

    with pytest.raises(DataReadinessError, match="outside matching source collection window"):
        publish_global_event_authority(
            pd.DataFrame({"decision_time_utc": [DECISION_TIME]}),
            [event_path],
            [coverage_path],
            tmp_path / "authority",
            required_historical_sources=("alpaca",),
            production_ready=False,
        )


def test_rejects_mixed_sentiment_scorer_identity(tmp_path: Path) -> None:
    event_path = _write_events(
        tmp_path / "events.parquet",
        _events(
            [_event("1", "alpaca", DECISION_TIME - pd.Timedelta(hours=1), 0.2)]
        ),
        production=False,
        scorer_identity="fixture-finbert|revision=v1|max_length=128",
    )
    coverage_path = _write_coverage(
        tmp_path / "coverage.parquet",
        _coverage(
            decisions=(DECISION_TIME,),
            sources=("alpaca",),
            alpaca_rows=(1,),
        ),
        production=False,
        scorer_identity="fixture-finbert|revision=v2|max_length=128",
    )

    with pytest.raises(DataReadinessError, match="sentiment scorer identity is inconsistent"):
        publish_global_event_authority(
            pd.DataFrame({"decision_time_utc": [DECISION_TIME]}),
            [event_path],
            [coverage_path],
            tmp_path / "authority",
            required_historical_sources=("alpaca",),
            production_ready=False,
        )


def test_rejects_missing_sentiment_scorer_identity(tmp_path: Path) -> None:
    event_path = _write_events(
        tmp_path / "events.parquet",
        _events(
            [_event("1", "alpaca", DECISION_TIME - pd.Timedelta(hours=1), 0.2)]
        ),
        production=False,
        scorer_identity=None,
    )
    coverage_path = _write_coverage(
        tmp_path / "coverage.parquet",
        _coverage(
            decisions=(DECISION_TIME,),
            sources=("alpaca",),
            alpaca_rows=(1,),
        ),
        production=False,
    )

    with pytest.raises(DataReadinessError, match="scorer_identity is missing"):
        publish_global_event_authority(
            pd.DataFrame({"decision_time_utc": [DECISION_TIME]}),
            [event_path],
            [coverage_path],
            tmp_path / "authority",
            required_historical_sources=("alpaca",),
            production_ready=False,
        )


def _event(
    suffix: str,
    source_family: str,
    feature_time: pd.Timestamp,
    sentiment: float,
    *,
    availability_policy: Literal["observed", "provider_publication_proxy"] = "observed",
    ticker: str = "MARKET",
    security_id: str = "market:global",
) -> dict[str, object]:
    published = feature_time - pd.Timedelta(minutes=5)
    first_seen = feature_time if availability_policy == "observed" else feature_time + pd.Timedelta(days=30)
    return cast(
        dict[str, object],
        CanonicalEvent(
            event_id=(suffix * 64)[:64],
            ticker=ticker,
            security_id=security_id,
            source_family=source_family,
            source=f"{source_family}:fixture",
            published_at_utc=published.to_pydatetime(),
            first_seen_at_utc=first_seen.to_pydatetime(),
            available_at_utc=feature_time.to_pydatetime(),
            sentiment_scored_at_utc=feature_time.to_pydatetime(),
            feature_available_at_utc=feature_time.to_pydatetime(),
            title=f"Global event {suffix}",
            sentiment_numeric=sentiment,
            relevance=1.0,
            availability_policy=availability_policy,
            raw_sha256=("f" + suffix * 63)[:64],
        ).model_dump(),
    )


def _events(records: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def _coverage(
    *,
    decisions: tuple[pd.Timestamp, ...],
    sources: tuple[str, ...],
    alpaca_rows: tuple[int, ...],
    collection_lag: pd.Timedelta = pd.Timedelta(0),
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for source in sources:
        for index, decision in enumerate(decisions):
            rows = alpaca_rows[index] if source == "alpaca" else 0
            records.append(
                SourceCollection(
                    collection_id=f"global-{source}-{index:04d}",
                    ticker="MARKET",
                    source_family=source,
                    requested_start_utc=(decision - pd.Timedelta(days=3)).to_pydatetime(),
                    requested_end_utc=decision.to_pydatetime(),
                    started_at_utc=(decision + collection_lag - pd.Timedelta(minutes=1)).to_pydatetime(),
                    completed_at_utc=(decision + collection_lag).to_pydatetime(),
                    status="observed" if rows else "observed_empty",
                    row_count=rows,
                ).model_dump()
            )
    return pd.DataFrame.from_records(records)


def _write_events(
    path: Path,
    frame: pd.DataFrame,
    *,
    production: bool,
    collection_request_sha256: str = COLLECTION_REQUEST_SHA256,
    source_policy_sha256: str = SOURCE_POLICY_SHA256,
    scorer_identity: str | None = SCORER_IDENTITY,
) -> Path:
    inputs: dict[str, str] = {
        "collection_request_sha256": collection_request_sha256,
        "source_policy_sha256": source_policy_sha256,
        "fixture_raw_sha256": "1" * 64,
    }
    if scorer_identity is not None:
        inputs["sentiment_scorer_identity"] = scorer_identity
    write_canonical_artifact(
        frame,
        path,
        artifact_type="events",
        audit=_passing_audit("events", len(frame)),
        inputs=inputs,
        production_ready=production,
    )
    return path


def _write_coverage(
    path: Path,
    frame: pd.DataFrame,
    *,
    production: bool,
    collection_request_sha256: str = COLLECTION_REQUEST_SHA256,
    source_policy_sha256: str = SOURCE_POLICY_SHA256,
    scorer_identity: str = SCORER_IDENTITY,
) -> Path:
    write_canonical_artifact(
        frame,
        path,
        artifact_type="source_collections",
        audit=_passing_audit("coverage", len(frame)),
        inputs={
            "collection_request_sha256": collection_request_sha256,
            "source_policy_sha256": source_policy_sha256,
            "sentiment_scorer_identity": scorer_identity,
            "fixture_raw_sha256": "2" * 64,
        },
        production_ready=production,
    )
    return path


def _passing_audit(name: str, rows: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name=name,
                status="pass",
                failures=0,
                rows_checked=rows,
                detail="fixture",
            ),
        )
    )
