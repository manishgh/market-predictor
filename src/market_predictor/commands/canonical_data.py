from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import typer

from market_predictor.canonical.audits import (
    CanonicalAuditReport,
    audit_canonical_bars,
    audit_canonical_events,
    audit_decision_availability,
    audit_decision_source_coverage,
    audit_fundamental_facts,
    audit_source_collections,
    audit_universe_memberships,
)
from market_predictor.canonical.contracts import AvailabilityPolicy
from market_predictor.canonical.joins import (
    aggregate_event_features,
    decisions_from_completed_bars,
    join_fundamentals_asof,
    join_source_collection_status,
    join_universe_membership,
)
from market_predictor.canonical.normalize import (
    canonicalize_bars,
    canonicalize_events,
    canonicalize_universe_memberships,
)
from market_predictor.canonical.store import file_sha256, load_canonical_artifact, write_canonical_artifact


def register_canonical_data_commands(app: typer.Typer, console: Any) -> None:
    @app.command("canonicalize-bars")
    def canonicalize_bars_command(
        input_path: Path = typer.Option(..., help="Raw OHLCV CSV or parquet."),
        out: Path = typer.Option(..., help="Canonical bar parquet."),
        timeframe: str | None = typer.Option(None, help="Required when input has no timeframe column."),
        ticker: str | None = typer.Option(None, help="Required when input has no ticker/symbol column."),
        price_feed: str | None = typer.Option(None, help="Required when input has no price_feed column."),
        availability_policy: str = typer.Option("market_interval_close", help="market_interval_close or observed."),
        intraday_finalization_seconds: int = typer.Option(30, min=0),
        daily_finalization_minutes: int = typer.Option(15, min=0),
        production: bool = typer.Option(True, "--production/--research"),
    ) -> None:
        """Normalize left-edge bars and publish an audited immutable artifact."""

        raw = _read_frame(input_path)
        bars = canonicalize_bars(
            raw,
            timeframe=timeframe,
            ticker=ticker,
            price_feed=price_feed,
            availability_policy=_bar_policy(availability_policy),
            intraday_finalization_delay=pd.Timedelta(seconds=intraday_finalization_seconds),
            daily_finalization_delay=pd.Timedelta(minutes=daily_finalization_minutes),
        )
        audit = CanonicalAuditReport(checks=audit_canonical_bars(bars, require_sip=production))
        manifest = write_canonical_artifact(
            bars,
            out,
            artifact_type="bars",
            audit=audit,
            inputs={str(input_path): file_sha256(input_path)},
            production_ready=production,
        )
        console.print({"rows": len(bars), "out": str(out), "sha256": manifest["artifact_sha256"]})

    @app.command("canonicalize-events")
    def canonicalize_events_command(
        input_path: Path = typer.Option(..., help="Raw event CSV or parquet."),
        out: Path = typer.Option(..., help="Canonical event parquet."),
        availability_policy: str = typer.Option("observed", help="observed or provider_publication_proxy."),
        collected_at_utc: str | None = typer.Option(
            None,
            help="Timezone-aware fallback only when rows lack ingestion timestamps.",
        ),
        sentiment_scored_at_utc: str | None = typer.Option(
            None,
            help="Timezone-aware fallback for historical sentiment rows lacking scoring timestamps.",
        ),
        production: bool = typer.Option(True, "--production/--research"),
    ) -> None:
        """Normalize event availability and publish an audited immutable artifact."""

        raw = _read_frame(input_path)
        policy = _event_policy(availability_policy)
        if production and policy != "observed":
            raise typer.BadParameter("production event artifacts require observed availability")
        events = canonicalize_events(
            raw,
            collected_at_utc=_optional_datetime(collected_at_utc),
            sentiment_scored_at_utc=_optional_datetime(sentiment_scored_at_utc),
            availability_policy=policy,
        )
        audit = CanonicalAuditReport(checks=audit_canonical_events(events, require_observed=production))
        manifest = write_canonical_artifact(
            events,
            out,
            artifact_type="events",
            audit=audit,
            inputs={str(input_path): file_sha256(input_path)},
            production_ready=production,
        )
        console.print({"rows": len(events), "out": str(out), "sha256": manifest["artifact_sha256"]})

    @app.command("canonicalize-source-collections")
    def canonicalize_source_collections_command(
        input_path: Path = typer.Option(..., help="Raw source collection CSV or parquet."),
        out: Path = typer.Option(..., help="Canonical source collection parquet."),
        production: bool = typer.Option(True, "--production/--research"),
    ) -> None:
        """Validate source attempt states and publish an immutable collection artifact."""

        collections = _read_frame(input_path)
        audit = CanonicalAuditReport(
            checks=audit_source_collections(collections, require_success=production)
        )
        manifest = write_canonical_artifact(
            collections,
            out,
            artifact_type="source_collections",
            audit=audit,
            inputs={str(input_path): file_sha256(input_path)},
            production_ready=production,
        )
        console.print({"rows": len(collections), "out": str(out), "sha256": manifest["artifact_sha256"]})

    @app.command("canonicalize-event-directory")
    def canonicalize_event_directory_command(
        input_dir: Path = typer.Option(..., help="Directory containing per-ticker *_events.parquet files."),
        out: Path = typer.Option(..., help="Combined canonical event parquet."),
        availability_policy: str = typer.Option("observed", help="observed or provider_publication_proxy."),
        collected_at_utc: str | None = typer.Option(None, help="Fallback for files without ingestion timestamps."),
        sentiment_scored_at_utc: str | None = typer.Option(None, help="Fallback for files without score timestamps."),
        production: bool = typer.Option(True, "--production/--research"),
    ) -> None:
        """Canonicalize and combine isolated per-ticker event files."""

        if not input_dir.is_dir():
            raise typer.BadParameter(f"input directory does not exist: {input_dir}")
        paths = sorted(input_dir.glob("*_events.parquet"))
        if not paths:
            raise typer.BadParameter(f"no *_events.parquet files found in {input_dir}")
        policy = _event_policy(availability_policy)
        if production and policy != "observed":
            raise typer.BadParameter("production event artifacts require observed availability")
        parts = [
            canonicalize_events(
                pd.read_parquet(path),
                collected_at_utc=_optional_datetime(collected_at_utc),
                sentiment_scored_at_utc=_optional_datetime(sentiment_scored_at_utc),
                availability_policy=policy,
            )
            for path in paths
        ]
        events = pd.concat(parts, ignore_index=True).sort_values(["feature_available_at_utc", "ticker"])
        if bool(events["event_id"].duplicated().any()):
            events = events.drop_duplicates("event_id", keep="first").reset_index(drop=True)
        audit = CanonicalAuditReport(checks=audit_canonical_events(events, require_observed=production))
        manifest = write_canonical_artifact(
            events,
            out,
            artifact_type="events",
            audit=audit,
            inputs={str(path): file_sha256(path) for path in paths},
            production_ready=production,
        )
        console.print(
            {"files": len(paths), "rows": len(events), "out": str(out), "sha256": manifest["artifact_sha256"]}
        )

    @app.command("canonicalize-fundamentals")
    def canonicalize_fundamentals_command(
        input_path: Path = typer.Option(..., help="Canonical SEC fact rows in CSV or parquet."),
        out: Path = typer.Option(..., help="Canonical fundamental fact parquet."),
        production: bool = typer.Option(True, "--production/--research"),
    ) -> None:
        """Validate versioned filing facts and publish an immutable fact artifact."""

        facts = _read_frame(input_path)
        audit = CanonicalAuditReport(checks=audit_fundamental_facts(facts, require_observed=production))
        manifest = write_canonical_artifact(
            facts,
            out,
            artifact_type="fundamentals",
            audit=audit,
            inputs={str(input_path): file_sha256(input_path)},
            production_ready=production,
        )
        console.print({"rows": len(facts), "out": str(out), "sha256": manifest["artifact_sha256"]})

    @app.command("canonicalize-memberships")
    def canonicalize_memberships_command(
        input_path: Path = typer.Option(..., help="Observed point-in-time membership CSV or parquet."),
        out: Path = typer.Option(..., help="Canonical universe membership parquet."),
        source: str | None = typer.Option(None, help="Required when input has no source column."),
        availability_policy: str | None = typer.Option(
            None,
            help="Required when input has no availability_policy column.",
        ),
        production: bool = typer.Option(True, "--production/--research"),
    ) -> None:
        """Publish observed universe history with explicit snapshot availability."""

        policy = _optional_availability_policy(availability_policy)
        if production and policy is not None and policy != "observed":
            raise typer.BadParameter("production membership artifacts require observed availability")
        memberships = canonicalize_universe_memberships(
            _read_frame(input_path),
            source=source,
            availability_policy=policy,
        )
        audit = CanonicalAuditReport(
            checks=audit_universe_memberships(memberships, require_observed=production)
        )
        manifest = write_canonical_artifact(
            memberships,
            out,
            artifact_type="memberships",
            audit=audit,
            inputs={str(input_path): file_sha256(input_path)},
            production_ready=production,
        )
        console.print({"rows": len(memberships), "out": str(out), "sha256": manifest["artifact_sha256"]})

    @app.command("build-canonical-decisions")
    def build_canonical_decisions_command(
        bars: Path = typer.Option(..., help="Hash-verified canonical bars."),
        events: Path = typer.Option(..., help="Hash-verified canonical events."),
        source_collections: Path = typer.Option(..., help="Source collection status CSV or parquet."),
        memberships: Path = typer.Option(..., help="Hash-verified point-in-time universe memberships."),
        out: Path = typer.Option(..., help="Canonical decision table parquet."),
        fundamentals: Path | None = typer.Option(None, help="Optional canonical fundamental facts."),
        fundamental_metrics: str = typer.Option("", help="Comma-separated metrics to join as of decision time."),
        required_sources: str = typer.Option(
            "alpaca,reddit,seeking_alpha,sec,finviz",
            help="Sources that must be observed for every ticker in production.",
        ),
        production: bool = typer.Option(True, "--production/--research"),
    ) -> None:
        """Build a fail-closed point-in-time decision table from canonical inputs."""

        bar_frame, _ = load_canonical_artifact(bars, expected_type="bars", allow_research=not production)
        event_frame, _ = load_canonical_artifact(events, expected_type="events", allow_research=not production)
        collection_frame, _ = load_canonical_artifact(
            source_collections,
            expected_type="source_collections",
            allow_research=not production,
        )
        membership_frame, _ = load_canonical_artifact(
            memberships,
            expected_type="memberships",
            allow_research=not production,
        )
        sources = _csv(required_sources)
        decisions = decisions_from_completed_bars(bar_frame)
        decisions = join_universe_membership(decisions, membership_frame)
        decisions = aggregate_event_features(decisions, event_frame, require_observed=production)
        decisions = join_source_collection_status(decisions, collection_frame, source_families=sources)
        feature_timestamps = [
            "feature_available_at_utc",
            "membership_available_at_utc",
            "latest_event_feature_available_at_utc",
            *(f"source_status_available_at_utc_{source}" for source in sources),
        ]
        checks = [
            *audit_canonical_bars(bar_frame, require_sip=production),
            *audit_canonical_events(event_frame, require_observed=production),
            *audit_source_collections(
                collection_frame,
                required_tickers=decisions["ticker"].astype(str).unique(),
                required_sources=sources if production else (),
                require_success=production,
            ),
            *audit_universe_memberships(
                membership_frame,
                decisions=decisions,
                require_observed=production,
            ),
        ]
        inputs = {
            str(bars): file_sha256(bars),
            str(events): file_sha256(events),
            str(source_collections): file_sha256(source_collections),
            str(memberships): file_sha256(memberships),
        }
        metrics = _csv(fundamental_metrics)
        if fundamentals is not None:
            fact_frame, _ = load_canonical_artifact(
                fundamentals,
                expected_type="fundamentals",
                allow_research=not production,
            )
            decisions = join_fundamentals_asof(
                decisions,
                fact_frame,
                metrics=metrics,
                require_observed=production,
            )
            checks.extend(audit_fundamental_facts(fact_frame, require_observed=production))
            feature_timestamps.extend(f"fundamental_available_at_utc_{metric}" for metric in metrics)
            inputs[str(fundamentals)] = file_sha256(fundamentals)
        checks.extend(audit_decision_availability(decisions, feature_timestamp_columns=feature_timestamps))
        if production:
            checks.extend(audit_decision_source_coverage(decisions, required_sources=sources))
        audit = CanonicalAuditReport(checks=tuple(checks))
        manifest = write_canonical_artifact(
            decisions,
            out,
            artifact_type="decisions",
            audit=audit,
            inputs=inputs,
            production_ready=production,
        )
        console.print({"rows": len(decisions), "out": str(out), "sha256": manifest["artifact_sha256"]})


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"input file does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise typer.BadParameter(f"unsupported input format: {path}")


def _optional_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise typer.BadParameter("timestamps must include a timezone")
    return cast(datetime, timestamp.to_pydatetime())


def _csv(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _bar_policy(value: str) -> AvailabilityPolicy:
    normalized = value.strip().lower()
    if normalized not in {"market_interval_close", "observed"}:
        raise typer.BadParameter("bar availability policy must be market_interval_close or observed")
    return cast(AvailabilityPolicy, normalized)


def _event_policy(value: str) -> AvailabilityPolicy:
    normalized = value.strip().lower()
    if normalized not in {"observed", "provider_publication_proxy"}:
        raise typer.BadParameter("event availability policy must be observed or provider_publication_proxy")
    return cast(AvailabilityPolicy, normalized)


def _optional_availability_policy(value: str | None) -> AvailabilityPolicy | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in {"observed", "provider_publication_proxy"}:
        raise typer.BadParameter("availability policy must be observed or provider_publication_proxy")
    return cast(AvailabilityPolicy, normalized)
