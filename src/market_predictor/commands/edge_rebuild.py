"""Edge-rebuild research audit commands."""

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from market_predictor.config import get_settings
from market_predictor.edge_rebuild.benchmark_history import (
    build_selected_session_benchmark_plan,
)
from market_predictor.edge_rebuild.broad_intraday_history import (
    build_broad_intraday_history_plan,
)
from market_predictor.edge_rebuild.catalyst_authority import (
    publish_catalyst_decision_authority,
)
from market_predictor.edge_rebuild.catalyst_identity_rebind import (
    publish_catalyst_identity_rebind,
)
from market_predictor.edge_rebuild.contracts import (
    load_edge_rebuild_readiness_config,
)
from market_predictor.edge_rebuild.extended_session_context import (
    build_extended_session_context_plan,
)
from market_predictor.edge_rebuild.global_event_authority import (
    publish_global_event_authority,
)
from market_predictor.edge_rebuild.global_event_collection import (
    GLOBAL_EVENT_QUERY_POLICY_V1,
    GdeltCollectionRequest,
    collect_live_gdelt_global_events,
    validate_gdelt_collection_request,
)
from market_predictor.edge_rebuild.history_collection import (
    collect_intraday_history,
)
from market_predictor.edge_rebuild.history_contracts import (
    load_broad_intraday_history_config,
    load_collection_transport_config,
    load_extended_session_context_config,
    load_intraday_history_config,
    load_selected_session_benchmark_config,
    load_selected_session_history_config,
    load_selected_session_one_minute_config,
)
from market_predictor.edge_rebuild.history_materialization import (
    reorganize_intraday_history,
)
from market_predictor.edge_rebuild.intraday_dataset import publish_intraday_dataset
from market_predictor.edge_rebuild.intraday_development import (
    evaluate_future_intraday_holdout,
    load_intraday_development_config,
    train_intraday_development_candidate,
)
from market_predictor.edge_rebuild.intraday_history import (
    build_intraday_history_plan,
)
from market_predictor.edge_rebuild.intraday_rejection import (
    publish_intraday_candidate_rejection,
)
from market_predictor.edge_rebuild.intraday_selection import (
    build_intraday_selection,
    publish_intraday_selection,
)
from market_predictor.edge_rebuild.one_minute_coverage import (
    publish_selected_session_one_minute_coverage,
)
from market_predictor.edge_rebuild.readiness import (
    run_edge_rebuild_readiness_audit,
)
from market_predictor.edge_rebuild.sec_filing_authority import (
    publish_sec_filing_decision_authority,
)
from market_predictor.edge_rebuild.sec_filing_collection import (
    collect_historical_sec_filings,
    load_sec_filing_collection_config,
    load_sec_identity_relations,
)
from market_predictor.edge_rebuild.sec_identity_authority import (
    load_sec_identity_config,
    publish_sec_identity_authority,
)
from market_predictor.edge_rebuild.selected_session_history import (
    build_selected_session_history_plan,
)
from market_predictor.edge_rebuild.sp500_memberships import (
    publish_sp500_membership_authority,
)
from market_predictor.edge_rebuild.sp500_transitions import (
    publish_sp500_transition_authority,
)
from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract
from market_predictor.edge_rebuild.swing_history_acquisition import (
    publish_swing_history_acquisition_plan,
)
from market_predictor.edge_rebuild.swing_materialization import (
    materialize_swing_feature_panel,
)
from market_predictor.edge_rebuild.swing_ordering import audit_swing_ordering
from market_predictor.edge_rebuild.swing_training import (
    load_swing_training_config,
    train_swing_edge_candidate,
)
from market_predictor.edge_rebuild.temporal_manifest import (
    load_temporal_manifest_config,
    publish_temporal_manifest,
)
from market_predictor.edge_rebuild.universe_identity import (
    publish_verified_universe,
)
from market_predictor.heavy_jobs import serialized_heavy_job
from market_predictor.intraday.specialist_contracts import (
    load_intraday_specialist_research_config,
)
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.sources.sec import SecRequestGovernor, SecSource
from market_predictor.v3.errors import DataReadinessError


def _iso_date(value: str, *, option: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter("expected YYYY-MM-DD", param_hint=option) from exc


def _iso_datetime(value: str, *, option: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise typer.BadParameter(
            "expected a timezone-aware ISO-8601 timestamp",
            param_hint=option,
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise typer.BadParameter(
            "expected a timezone-aware ISO-8601 timestamp",
            param_hint=option,
        )
    return parsed.astimezone(UTC)


def register_edge_rebuild_commands(app: typer.Typer, console: Any) -> None:
    @app.command("publish-edge-sec-identity-authority")
    @serialized_heavy_job("publish-edge-sec-identity-authority")
    def publish_edge_sec_identity_authority(
        sec_mapping: Path = typer.Option(
            ...,
            help="Existing local raw SEC company_tickers.json; never downloaded.",
        ),
        reviewed_overrides: Path = typer.Option(
            Path("configs/sec_identity_reviewed_overrides.csv"),
            help="Reviewed, evidence-backed effective-dated SEC identity overrides.",
        ),
        membership_dir: Path = typer.Option(...),
        raw_archive_dir: Path = typer.Option(...),
        event_authority_dir: Path = typer.Option(...),
        transition_authority_dir: Path = typer.Option(...),
        reviewed_transitions: Path = typer.Option(...),
        anchor: Path = typer.Option(...),
        out_dir: Path = typer.Option(..., help="New immutable SEC identity authority."),
        policy: Path = typer.Option(Path("configs/edge_rebuild_sec_identity.toml")),
    ) -> None:
        """Publish the offline effective-dated SEC CIK-to-security relation."""

        result = publish_sec_identity_authority(
            sec_mapping_path=sec_mapping,
            reviewed_overrides_path=reviewed_overrides,
            membership_directory=membership_dir,
            archive_directory=raw_archive_dir,
            event_directory=event_authority_dir,
            transition_directory=transition_authority_dir,
            reviewed_transitions_path=reviewed_transitions,
            anchor_path=anchor,
            output_directory=out_dir,
            config=load_sec_identity_config(policy),
        )
        coverage = result.manifest.get("coverage")
        if not isinstance(coverage, dict):
            raise DataReadinessError("SEC identity authority has no coverage summary")
        console.print(
            {
                "status": "complete",
                "securities": coverage["security_count"],
                "excluded_securities": coverage["excluded_security_count"],
                "issuer_ciks": coverage["issuer_cik_count"],
                "directory": str(result.directory),
            }
        )

    @app.command("collect-edge-sec-filings")
    @serialized_heavy_job("collect-edge-sec-filings")
    def collect_edge_sec_filings(
        identity_relations: Path = typer.Option(
            ...,
            help="Effective-dated security-to-CIK relation Parquet or CSV.",
        ),
        out_dir: Path = typer.Option(..., help="New immutable SEC collection directory."),
        policy: Path = typer.Option(Path("configs/edge_rebuild_sec_filings.toml")),
    ) -> None:
        """Collect historical issuer filings with research-only availability lineage."""

        collection_config = load_sec_filing_collection_config(policy)
        relations = load_sec_identity_relations(identity_relations)
        settings = get_settings()
        governor = SecRequestGovernor(
            requests_per_second=collection_config.requests_per_second,
            forbidden_cooldown_seconds=collection_config.forbidden_cooldown_seconds,
            rate_limit_cooldown_seconds=collection_config.rate_limit_cooldown_seconds,
        )
        result = collect_historical_sec_filings(
            relations,
            out_dir,
            source_factory=lambda: SecSource(settings, governor=governor),
            config=collection_config,
        )
        console.print(
            {
                "status": "complete",
                "issuers": result.manifest["issuer_count"],
                "failed_issuers": result.manifest["failed_issuers"],
                "events": len(result.events),
                "production_ready": False,
                "directory": str(result.directory),
            }
        )

    @app.command("publish-edge-sec-filing-authority")
    @serialized_heavy_job("publish-edge-sec-filing-authority")
    def publish_edge_sec_filing_authority(
        decisions: Path = typer.Option(..., help="Canonical swing decisions Parquet."),
        collection_dir: list[Path] = typer.Option(
            ...,
            "--collection-dir",
            help="Immutable SEC collection directory; repeat for each generation.",
        ),
        identity_relations: Path = typer.Option(
            ...,
            help="Effective-dated security-to-CIK relation Parquet or CSV.",
        ),
        out_dir: Path = typer.Option(..., help="New immutable research authority directory."),
    ) -> None:
        """Publish research-only accepted-time SEC filing features."""

        result = publish_sec_filing_decision_authority(
            decisions,
            collection_dir,
            identity_relations,
            out_dir,
            production_ready=False,
        )
        console.print(
            {
                "status": "complete",
                "decisions": result.decision_rows,
                "coverage_rows": len(result.coverage),
                "production_ready": False,
                "directory": str(result.directory),
            }
        )

    @app.command("publish-edge-catalyst-authority")
    @serialized_heavy_job("publish-edge-catalyst-authority")
    def publish_edge_catalyst_authority(
        lineage_dir: list[Path] = typer.Option(
            ...,
            "--lineage-dir",
            help="Completed catalyst-lineage directory; repeat for each generation.",
        ),
        out_dir: Path = typer.Option(..., help="New immutable authority directory."),
        production_ready: bool = typer.Option(False),
    ) -> None:
        """Merge verified catalyst generations into one decision authority."""

        result = publish_catalyst_decision_authority(
            lineage_dir,
            out_dir,
            production_ready=production_ready,
        )
        console.print(
            {
                "status": "complete",
                "decisions": len(result.decisions),
                "coverage_rows": len(result.coverage),
                "production_ready": result.manifest["production_ready"],
                "directory": str(result.directory),
            }
        )

    @app.command("publish-edge-catalyst-identity-rebind")
    @serialized_heavy_job("publish-edge-catalyst-identity-rebind")
    def publish_edge_catalyst_identity_rebind(
        parent_dir: Path = typer.Option(
            ...,
            help="Verified legacy catalyst decision authority to rebind.",
        ),
        target_panel_dir: Path = typer.Option(
            ...,
            help="Verified technical swing panel defining canonical decisions.",
        ),
        out_dir: Path = typer.Option(..., help="New immutable rebind authority directory."),
    ) -> None:
        """Rebind catalyst evidence to canonical point-in-time security identity."""

        result = publish_catalyst_identity_rebind(
            parent_directory=parent_dir,
            target_panel_directory=target_panel_dir,
            output_directory=out_dir,
        )
        console.print(
            {
                "status": "complete",
                "decisions": len(result.decisions),
                "coverage_rows": len(result.coverage),
                "directory": str(result.directory),
            }
        )

    @app.command("collect-edge-live-global-context")
    @serialized_heavy_job("collect-edge-live-global-context")
    def collect_edge_live_global_context(
        start: str = typer.Option(..., help="Inclusive UTC publication start."),
        end: str = typer.Option(..., help="Inclusive UTC publication end."),
        out_dir: Path = typer.Option(..., help="New immutable collection directory."),
        max_records_per_query: int = typer.Option(250, min=1, max=250),
        sentiment_batch_size: int = typer.Option(64, min=1, max=250),
    ) -> None:
        """Collect one complete live GDELT query-policy window."""

        from market_predictor.sentiment import FinbertScorer

        try:
            request = validate_gdelt_collection_request(
                GdeltCollectionRequest(
                    queries=GLOBAL_EVENT_QUERY_POLICY_V1,
                    requested_start_utc=_iso_datetime(start, option="--start"),
                    requested_end_utc=_iso_datetime(end, option="--end"),
                    max_records=max_records_per_query,
                )
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if request.requested_end_utc > datetime.now(UTC):
            raise typer.BadParameter("--end cannot be in the future")
        settings = get_settings()
        scorer = FinbertScorer(
            settings.finbert_model,
            torch_num_threads=settings.torch_num_threads,
            max_length=128,
        )
        result = collect_live_gdelt_global_events(
            request,
            out_dir,
            scorer=scorer,
            scorer_batch_size=sentiment_batch_size,
            scorer_identity=(f"{scorer.model_name}|{scorer.model_revision}|max_length=128|device={scorer.device}"),
        )
        console.print(
            {
                "status": "complete",
                "events": len(result.events),
                "coverage": result.source_collections.iloc[0]["status"],
                "directory": str(result.directory),
            }
        )

    @app.command("publish-edge-global-event-authority")
    @serialized_heavy_job("publish-edge-global-event-authority")
    def publish_edge_global_event_authority(
        decisions: Path = typer.Option(..., help="Decision-time Parquet."),
        event_artifact: list[Path] = typer.Option(
            ...,
            "--event-artifact",
            help="Canonical global event Parquet; repeat for each collection.",
        ),
        coverage_artifact: list[Path] = typer.Option(
            ...,
            "--coverage-artifact",
            help="Canonical source-collection Parquet; repeat for each collection.",
        ),
        required_source: list[str] = typer.Option(
            ...,
            "--required-source",
            help="Required global source family; repeat for each family.",
        ),
        out_dir: Path = typer.Option(...),
        production_ready: bool = typer.Option(False),
    ) -> None:
        """Publish exact decision-time global features from verified collections."""

        result = publish_global_event_authority(
            pd.read_parquet(decisions, columns=["decision_time_utc"]),
            event_artifact,
            coverage_artifact,
            out_dir,
            required_historical_sources=required_source,
            production_ready=production_ready,
        )
        console.print(
            {
                "status": "complete",
                "decisions": len(result.decisions),
                "coverage_rows": len(result.coverage),
                "directory": str(result.directory),
            }
        )

    @app.command("plan-edge-rebuild-broad-intraday-history")
    @serialized_heavy_job("plan-edge-rebuild-broad-intraday-history")
    def plan_edge_rebuild_broad_intraday_history(
        broad_memberships: Path = typer.Option(...),
        pit_memberships: Path = typer.Option(...),
        existing_corpus_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        policy: Path = typer.Option(Path("configs/edge_rebuild_broad_intraday_history.toml")),
    ) -> None:
        """Plan missing causal five-minute history for the broad universe."""

        result = build_broad_intraday_history_plan(
            broad_memberships_path=broad_memberships,
            pit_memberships_path=pit_memberships,
            existing_corpus_directory=existing_corpus_dir,
            policy_path=policy,
            output_directory=out_dir,
            config=load_broad_intraday_history_config(policy),
        )
        console.print(result["summary"])

    @app.command("publish-edge-rebuild-sp500-transitions")
    @serialized_heavy_job("publish-edge-rebuild-sp500-transitions")
    def publish_edge_rebuild_sp500_transitions(
        archive_dir: Path = typer.Option(..., help="Verified official S&P raw archive."),
        event_dir: Path = typer.Option(..., help="Verified offline S&P event authority."),
        reviewed_transitions: Path = typer.Option(
            Path("configs/sp500_security_transition_review.csv"),
            help="Reviewed security-transition ledger.",
        ),
        start_date: str = typer.Option(..., help="Inclusive YYYY-MM-DD start date."),
        cutoff_date: str = typer.Option(..., help="Inclusive YYYY-MM-DD cutoff date."),
        out_dir: Path = typer.Option(..., help="New immutable transition authority directory."),
    ) -> None:
        """Publish independent S&P ticker-transition authority offline."""

        result = publish_sp500_transition_authority(
            archive_directory=archive_dir,
            event_directory=event_dir,
            reviewed_transitions_path=reviewed_transitions,
            start_date=_iso_date(start_date, option="--start-date"),
            cutoff_date=_iso_date(cutoff_date, option="--cutoff-date"),
            output_directory=out_dir,
        )
        console.print(
            {
                "status": result["status"],
                "transitions": result["transition_count"],
                "transition_set_sha256": result["transition_set_sha256"],
                "out_dir": str(out_dir),
            }
        )

    @app.command("publish-edge-rebuild-sp500-memberships")
    @serialized_heavy_job("publish-edge-rebuild-sp500-memberships")
    def publish_edge_rebuild_sp500_memberships(
        archive_dir: Path = typer.Option(..., help="Verified official S&P raw archive."),
        event_dir: Path = typer.Option(..., help="Verified offline S&P event authority."),
        transition_dir: Path = typer.Option(..., help="Verified S&P transition authority."),
        anchor: Path = typer.Option(..., help="Cutoff-date S&P constituent anchor CSV."),
        reviewed_transitions: Path = typer.Option(
            Path("configs/sp500_security_transition_review.csv"),
            help="Reviewed security-transition ledger bound by transition authority.",
        ),
        start_date: str = typer.Option(..., help="Inclusive YYYY-MM-DD start date."),
        cutoff_date: str = typer.Option(..., help="Inclusive YYYY-MM-DD cutoff date."),
        out_dir: Path = typer.Option(..., help="New immutable membership authority directory."),
        security_exclusions: Path | None = typer.Option(
            None,
            help="Optional whole-security exclusion ledger.",
        ),
        maximum_security_exclusion_fraction: float = typer.Option(
            0.05,
            min=0.0,
            max=0.05,
        ),
    ) -> None:
        """Reconstruct anchor-bound point-in-time S&P membership offline."""

        result = publish_sp500_membership_authority(
            archive_directory=archive_dir,
            event_directory=event_dir,
            transition_directory=transition_dir,
            reviewed_transitions_path=reviewed_transitions,
            anchor_path=anchor,
            start_date=_iso_date(start_date, option="--start-date"),
            cutoff_date=_iso_date(cutoff_date, option="--cutoff-date"),
            output_directory=out_dir,
            security_exclusions_path=security_exclusions,
            maximum_security_exclusion_fraction=maximum_security_exclusion_fraction,
        )
        console.print(
            {
                "status": result["status"],
                "membership_intervals": result["membership_intervals"],
                "securities": result["security_count"],
                "excluded_securities": result["excluded_security_count"],
                "universe_sha256": result["universe_sha256"],
                "out_dir": str(out_dir),
            }
        )

    @app.command("plan-edge-rebuild-swing-history")
    @serialized_heavy_job("plan-edge-rebuild-swing-history")
    def plan_edge_rebuild_swing_history(
        temporal_manifest_dir: Path = typer.Option(
            ...,
            help="Verified temporal gap authority directory.",
        ),
        membership_authority_dir: Path = typer.Option(
            ...,
            help="Verified 2018-2026 PIT membership authority directory.",
        ),
        raw_archive_dir: Path = typer.Option(
            ...,
            help="Raw S&P archive bound by the membership authority.",
        ),
        event_authority_dir: Path = typer.Option(
            ...,
            help="S&P event authority bound by the membership authority.",
        ),
        transition_authority_dir: Path = typer.Option(
            ...,
            help="S&P transition authority bound by the membership authority.",
        ),
        reviewed_transitions: Path = typer.Option(
            ...,
            help="Reviewed transition ledger bound by the authority.",
        ),
        anchor: Path = typer.Option(
            ...,
            help="Cutoff constituent anchor bound by the authority.",
        ),
        current_daily_collection_dir: Path = typer.Option(
            ...,
            help="Existing verified daily-bar collection directory.",
        ),
        out_dir: Path = typer.Option(
            ...,
            help="New immutable acquisition-plan directory.",
        ),
        security_exclusions: Path | None = typer.Option(
            None,
            help="Optional whole-security exclusion ledger used by the authority.",
        ),
        repository_root: Path = typer.Option(
            Path("."),
            help="Repository boundary for every input and output.",
        ),
    ) -> None:
        """Plan missing swing history from the fully verified PIT authority."""

        result = publish_swing_history_acquisition_plan(
            repository_root=repository_root,
            temporal_manifest_directory=temporal_manifest_dir,
            membership_authority_directory=membership_authority_dir,
            raw_archive_directory=raw_archive_dir,
            event_authority_directory=event_authority_dir,
            transition_authority_directory=transition_authority_dir,
            reviewed_transitions_path=reviewed_transitions,
            anchor_path=anchor,
            current_daily_collection_directory=current_daily_collection_dir,
            output_directory=out_dir,
            security_exclusions_path=security_exclusions,
        )
        membership = result["membership"]
        console.print(
            {
                "status": result["status"],
                "missing_session_ranges": result["missing_session_ranges"],
                "membership": {
                    key: membership[key]
                    for key in (
                        "current_membership_start",
                        "authority_cutoff",
                        "required_start",
                        "required_end",
                        "security_count",
                        "excluded_security_count",
                        "universe_sha256",
                    )
                },
                "daily_bars": result["daily_bars"],
            }
        )

    @app.command("freeze-edge-rebuild-temporal-manifest")
    @serialized_heavy_job("freeze-edge-rebuild-temporal-manifest")
    def freeze_edge_rebuild_temporal_manifest(
        panel_dir: Path = typer.Option(
            ...,
            help="Published swing panel used only for provenance and session coverage.",
        ),
        out_dir: Path = typer.Option(..., help="New immutable manifest directory."),
        policy: Path = typer.Option(Path("configs/edge_rebuild_temporal_manifest.toml")),
        contract: Path = typer.Option(Path("configs/edge_rebuild_strategy_contract.toml")),
    ) -> None:
        """Freeze train, validation, embargo, holdout, and locked-test scopes."""

        result = publish_temporal_manifest(
            panel_directory=panel_dir,
            policy_path=policy,
            strategy_contract=load_strategy_contract(contract),
            output_directory=out_dir,
            config=load_temporal_manifest_config(policy),
        )
        console.print(
            {
                "status": result["status"],
                "target": result["target"],
                "locked_test": result["locked_test"],
                "coverage": result["coverage"],
            }
        )

    @app.command("audit-edge-rebuild-swing-ordering")
    @serialized_heavy_job("audit-edge-rebuild-swing-ordering")
    def audit_edge_rebuild_swing_ordering(
        panel_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        policy: Path = typer.Option(
            Path("configs/edge_rebuild_swing_ordering.toml"),
        ),
    ) -> None:
        """Test deterministic technical ordering before fitting a model."""

        result = audit_swing_ordering(
            panel_dir=panel_dir,
            config_path=policy,
            output_dir=out_dir,
        )
        console.print(
            {
                key: result[key]
                for key in (
                    "status",
                    "sessions",
                    "mean_session_spread_bps",
                    "positive_session_share",
                    "newey_west_t_stat",
                    "gates",
                )
            }
        )

    @app.command("materialize-edge-rebuild-swing-panel")
    @serialized_heavy_job("materialize-edge-rebuild-swing-panel")
    def materialize_edge_rebuild_swing_panel(
        pre_plan_dir: Path = typer.Option(
            ...,
            help="Verified exact pre-2019 acquisition-plan authority.",
        ),
        pre_collection_dir: Path = typer.Option(
            ...,
            help="Verified exact 2018-05-29 through 2019-07-08 collection.",
        ),
        post_collection_dir: Path = typer.Option(
            ...,
            help="Verified 2019-07-09 through 2026-07-08 collection.",
        ),
        membership_authority_dir: Path = typer.Option(
            ...,
            help="Verified 2018-2026 point-in-time membership authority.",
        ),
        raw_archive_dir: Path = typer.Option(
            ...,
            help="Raw S&P archive bound by the membership authority.",
        ),
        event_authority_dir: Path = typer.Option(
            ...,
            help="S&P event authority bound by the membership authority.",
        ),
        transition_authority_dir: Path = typer.Option(
            ...,
            help="S&P transition authority bound by the membership authority.",
        ),
        reviewed_transitions: Path = typer.Option(
            ...,
            help="Reviewed transition ledger bound by the authority.",
        ),
        anchor: Path = typer.Option(
            ...,
            help="Cutoff constituent anchor bound by the authority.",
        ),
        catalyst_authority_dir: Path = typer.Option(
            ...,
            help=("Verified decision-level catalyst authority; Alpaca history is required and optional source gaps remain explicit."),
        ),
        out_dir: Path = typer.Option(
            ...,
            help="New or matching resumable materialization directory.",
        ),
        contract: Path = typer.Option(
            Path("configs/edge_rebuild_strategy_contract.toml"),
        ),
        securities_per_shard: int = typer.Option(32, min=1),
        max_stage_one_shards: int | None = typer.Option(
            None,
            min=1,
            help="Optional resumable operational limit; stage two waits.",
        ),
        security_exclusions: Path | None = typer.Option(
            None,
            help="Optional exclusions already bound by the membership authority.",
        ),
    ) -> None:
        """Publish the complete 2018-2026 causal swing ranking panel."""

        result = materialize_swing_feature_panel(
            pre_plan_directory=pre_plan_dir,
            pre_collection_directory=pre_collection_dir,
            post_collection_directory=post_collection_dir,
            membership_directory=membership_authority_dir,
            raw_archive_directory=raw_archive_dir,
            event_directory=event_authority_dir,
            transition_directory=transition_authority_dir,
            reviewed_transitions_path=reviewed_transitions,
            anchor_path=anchor,
            catalyst_authority_directory=catalyst_authority_dir,
            contract=load_strategy_contract(contract),
            output_dir=out_dir,
            security_exclusions_path=security_exclusions,
            securities_per_shard=securities_per_shard,
            maximum_stage_one_shards_this_run=max_stage_one_shards,
        )
        console.print(
            {
                key: result[key]
                for key in (
                    "status",
                    "completed_stage_one_shards",
                    "total_stage_one_shards",
                    "rows",
                    "securities",
                    "sessions",
                )
                if key in result
            }
        )

    @app.command("publish-verified-universe")
    @serialized_heavy_job("publish-verified-universe")
    def publish_verified_universe_command(
        memberships: Path = typer.Option(...),
        daily_bars_dir: Path = typer.Option(
            ...,
            help="Daily bar corpus supplying identity evidence.",
        ),
        out: Path = typer.Option(...),
        audit_out: Path = typer.Option(...),
    ) -> None:
        """Publish only membership intervals whose symbol claim bar evidence supports."""

        import pandas as pd

        frames = []
        for path in sorted(daily_bars_dir.rglob("*.parquet")):
            frame = pd.read_parquet(
                path,
                columns=["ticker", "bar_start_utc", "close", "volume"],
            )
            # A zero-volume daily bar is a provider placeholder, not an observation.
            frame = frame[frame["volume"] > 0]
            frame["session"] = frame["bar_start_utc"].dt.tz_convert("America/New_York").dt.date.astype(str)
            frames.append(frame.rename(columns={"close": "last_close"}).assign(bars=1)[["session", "ticker", "bars", "last_close"]])
        if not frames:
            raise DataReadinessError(f"no daily bars found under {daily_bars_dir}")
        audit = publish_verified_universe(
            memberships_path=memberships,
            evidence=pd.concat(frames, ignore_index=True),
            output_path=out,
            audit_path=audit_out,
        )
        console.print(
            {
                key: audit[key]
                for key in (
                    "source_securities",
                    "kept_securities",
                    "excluded_security_share",
                    "excluded_intervals",
                    "unevaluated_intervals",
                )
            }
        )

    @app.command("materialize-edge-rebuild-intraday-history")
    @serialized_heavy_job("materialize-edge-rebuild-intraday-history")
    def materialize_edge_rebuild_intraday_history(
        regular_collection_dir: Path = typer.Option(...),
        extended_collection_dir: Path = typer.Option(...),
        legacy_stock_dir: Path = typer.Option(...),
        legacy_benchmark_dir: Path = typer.Option(...),
        universe: Path = typer.Option(
            ...,
            help="Verified canonical membership artifact.",
        ),
        out_dir: Path = typer.Option(...),
        first_session: str = typer.Option(..., help="Window start YYYY-MM-DD."),
        last_session: str = typer.Option(..., help="Window end YYYY-MM-DD."),
        selected_session_collection_dir: Path | None = typer.Option(
            None,
            help="Collected bars for the screened in-play stock-sessions.",
        ),
        selected_sessions: Path | None = typer.Option(
            None,
            help=("Published two-layer screen making those stock-sessions eligible; required with --selected-session-collection-dir."),
        ),
    ) -> None:
        """Reorganize downloaded bars into per-symbol regular and extended stores."""

        if (selected_session_collection_dir is None) != (selected_sessions is None):
            raise DataReadinessError("selected-session bars and their published screen must be supplied together")
        collected = {
            "regular": regular_collection_dir,
            "extended": extended_collection_dir,
        }
        if selected_session_collection_dir is not None:
            collected["selected_sessions"] = selected_session_collection_dir
        result = reorganize_intraday_history(
            collected_dirs=collected,
            legacy_dirs={
                "legacy_stocks": legacy_stock_dir,
                "legacy_benchmarks": legacy_benchmark_dir,
            },
            universe_path=universe,
            output_dir=out_dir,
            first_session=first_session,
            last_session=last_session,
            selected_sessions_path=selected_sessions,
        )
        console.print(
            {
                "symbols": result["symbols"],
                "total_rows": result["total_rows"],
                "rows_by_segment": result["rows_by_segment"],
                "window_sessions": result["window_sessions"],
                "defects": result["integrity"]["defect_count"],
            }
        )

    @app.command("plan-edge-rebuild-extended-session-context")
    @serialized_heavy_job("plan-edge-rebuild-extended-session-context")
    def plan_edge_rebuild_extended_session_context(
        intraday_plan_dir: Path = typer.Option(...),
        intraday_collection_dir: Path = typer.Option(...),
        memberships: Path = typer.Option(...),
        membership_audit: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        policy: Path = typer.Option(Path("configs/edge_rebuild_extended_session_context.toml")),
        first_session: str | None = typer.Option(
            None,
            help="Narrow the plan to a suffix of the frozen ER1A range (YYYY-MM-DD).",
        ),
    ) -> None:
        """Plan the separate ER1B pre/post-market five-minute context layer."""

        result = build_extended_session_context_plan(
            intraday_plan_directory=intraday_plan_dir,
            intraday_collection_directory=intraday_collection_dir,
            memberships_path=memberships,
            membership_audit_path=membership_audit,
            policy_path=policy,
            output_directory=out_dir,
            config=load_extended_session_context_config(policy),
            first_session=first_session,
        )
        console.print(result["summary"])

    @app.command("collect-edge-rebuild-intraday-history")
    @serialized_heavy_job("collect-edge-rebuild-intraday-history")
    def collect_edge_rebuild_intraday_history(
        plan_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        max_units: int | None = typer.Option(
            None,
            min=1,
            help="Optional resumable operational batch limit.",
        ),
        policy: Path = typer.Option(
            Path("configs/edge_rebuild_intraday_history.toml"),
            help=("Collection policy for the layer being collected. The layer comes from the policy's own schema_version."),
        ),
    ) -> None:
        """Collect resumable PIT SIP intraday bars for any planned layer."""

        settings = get_settings()
        config = load_collection_transport_config(policy)
        result = collect_intraday_history(
            plan_directory=plan_dir,
            policy_path=policy,
            output_directory=out_dir,
            config=config,
            source_factory=lambda: AlpacaSource(settings),
            maximum_units_this_run=max_units,
        )
        console.print(
            {
                "status": result["status"],
                "completed_units": result["completed_units"],
                "requested_units": result["requested_units"],
            }
        )

    @app.command("plan-edge-rebuild-selected-session-history")
    @serialized_heavy_job("plan-edge-rebuild-selected-session-history")
    def plan_edge_rebuild_selected_session_history(
        selection_dir: Path = typer.Option(
            ...,
            help="Published two-layer screen supplying the stock-sessions.",
        ),
        out_dir: Path = typer.Option(...),
        policy: Path = typer.Option(Path("configs/edge_rebuild_selected_session_history.toml")),
        contract: Path = typer.Option(Path("configs/edge_rebuild_strategy_contract.toml")),
    ) -> None:
        """Plan five-minute bars for exactly the selected in-play stock-sessions."""

        result = build_selected_session_history_plan(
            selection_directory=selection_dir,
            policy_path=policy,
            output_directory=out_dir,
            config=load_selected_session_history_config(policy),
            strategy_contract=load_strategy_contract(contract),
            strategy_contract_path=contract,
        )
        console.print(result["summary"])

    @app.command("plan-edge-rebuild-selected-session-one-minute")
    @serialized_heavy_job("plan-edge-rebuild-selected-session-one-minute")
    def plan_edge_rebuild_selected_session_one_minute(
        selection_dir: Path = typer.Option(
            ...,
            help="Published two-layer screen supplying the stock-sessions.",
        ),
        out_dir: Path = typer.Option(...),
        policy: Path = typer.Option(Path("configs/edge_rebuild_selected_session_one_minute.toml")),
        contract: Path = typer.Option(Path("configs/edge_rebuild_strategy_contract.toml")),
    ) -> None:
        """Plan one-minute bars for volume features and exact trade paths."""

        result = build_selected_session_history_plan(
            selection_directory=selection_dir,
            policy_path=policy,
            output_directory=out_dir,
            config=load_selected_session_one_minute_config(policy),
            strategy_contract=load_strategy_contract(contract),
            strategy_contract_path=contract,
        )
        console.print(result["summary"])

    @app.command("plan-edge-rebuild-selected-session-benchmarks")
    @serialized_heavy_job("plan-edge-rebuild-selected-session-benchmarks")
    def plan_edge_rebuild_selected_session_benchmarks(
        selection_dir: Path = typer.Option(
            ...,
            help="Published causal screen supplying the decision sessions.",
        ),
        out_dir: Path = typer.Option(...),
        policy: Path = typer.Option(Path("configs/edge_rebuild_selected_session_benchmarks.toml")),
        contract: Path = typer.Option(Path("configs/edge_rebuild_strategy_contract.toml")),
    ) -> None:
        """Plan one-minute SPY, QQQ, and sector-ETF paths."""

        result = build_selected_session_benchmark_plan(
            selection_directory=selection_dir,
            policy_path=policy,
            output_directory=out_dir,
            config=load_selected_session_benchmark_config(policy),
            strategy_contract=load_strategy_contract(contract),
            strategy_contract_path=contract,
        )
        console.print(result["summary"])

    @app.command("audit-edge-rebuild-selected-session-one-minute")
    @serialized_heavy_job("audit-edge-rebuild-selected-session-one-minute")
    def audit_edge_rebuild_selected_session_one_minute(
        plan_dir: Path = typer.Option(...),
        collection_dir: Path = typer.Option(...),
        five_minute_canonical_dir: Path = typer.Option(
            ...,
            help="Verified merged canonical regular-session five-minute store.",
        ),
        out_dir: Path = typer.Option(...),
        contract: Path = typer.Option(Path("configs/edge_rebuild_strategy_contract.toml")),
    ) -> None:
        """Publish stock-session coverage and whole-security exclusions."""

        result = publish_selected_session_one_minute_coverage(
            plan_directory=plan_dir,
            collection_directory=collection_dir,
            five_minute_canonical_directory=five_minute_canonical_dir,
            strategy_contract=load_strategy_contract(contract),
            strategy_contract_path=contract,
            output_directory=out_dir,
        )
        console.print(
            {
                "status": result["status"],
                "ready_for_feature_build": result["ready_for_feature_build"],
                **result["summary"],
            }
        )

    @app.command("publish-edge-rebuild-intraday-dataset")
    @serialized_heavy_job("publish-edge-rebuild-intraday-dataset")
    def publish_edge_rebuild_intraday_dataset(
        selection_dir: Path = typer.Option(...),
        stock_collection_dir: Path = typer.Option(...),
        stock_coverage_dir: Path = typer.Option(...),
        benchmark_collection_dir: Path = typer.Option(...),
        membership_authority_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        contract: Path = typer.Option(Path("configs/edge_rebuild_strategy_contract.toml")),
        session_workers: int = typer.Option(4, min=1, max=4),
    ) -> None:
        """Publish the immutable causal intraday feature and label dataset."""

        result = publish_intraday_dataset(
            selection_directory=selection_dir,
            stock_collection_directory=stock_collection_dir,
            stock_coverage_directory=stock_coverage_dir,
            benchmark_collection_directory=benchmark_collection_dir,
            membership_authority_directory=membership_authority_dir,
            strategy_contract=load_strategy_contract(contract),
            strategy_contract_path=contract,
            output_directory=out_dir,
            session_workers=session_workers,
        )
        console.print(result["summary"])

    @app.command("train-edge-rebuild-intraday-development")
    @serialized_heavy_job("train-edge-rebuild-intraday-development")
    def train_edge_rebuild_intraday_development(
        dataset_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        policy: Path = typer.Option(Path("configs/edge_rebuild_intraday_development.toml")),
    ) -> None:
        """Train only on data through 2026-07-08 and fail closed on economics."""

        result = train_intraday_development_candidate(
            dataset_authority_directory=dataset_dir,
            output_directory=out_dir,
            config=load_intraday_development_config(policy),
        )
        console.print(
            {
                "status": result.status,
                "selected_candidate_id": result.selected_candidate_id,
                "future_holdout_opened": False,
                "out_dir": str(result.output_directory),
            }
        )

    @app.command("evaluate-edge-rebuild-intraday-future-holdout")
    @serialized_heavy_job("evaluate-edge-rebuild-intraday-future-holdout")
    def evaluate_edge_rebuild_intraday_future_holdout(
        candidate_dir: Path = typer.Option(...),
        future_dataset_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
    ) -> None:
        """Open a post-2026-07-08 holdout only after development gates pass."""

        result = evaluate_future_intraday_holdout(candidate_dir, future_dataset_dir, out_dir)
        console.print({"status": result["status"], "out_dir": str(out_dir)})

    @app.command("publish-edge-rebuild-intraday-rejection")
    def publish_edge_rebuild_intraday_rejection(
        candidate_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        policy: Path = typer.Option(...),
    ) -> None:
        """Publish hash-bound rejection evidence without mutating the candidate."""

        result = publish_intraday_candidate_rejection(candidate_dir, policy, out_dir)
        console.print({"status": result["status"], "out_dir": str(out_dir)})

    @app.command("train-edge-rebuild-swing-candidate")
    @serialized_heavy_job("train-edge-rebuild-swing-candidate")
    def train_edge_rebuild_swing_candidate(
        panel_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        policy: Path = typer.Option(Path("configs/edge_rebuild_swing_training.toml")),
        contract: Path = typer.Option(Path("configs/edge_rebuild_strategy_contract.toml")),
        temporal_policy: Path = typer.Option(
            Path("configs/edge_rebuild_temporal_manifest.toml")
        ),
    ) -> None:
        """Train and evaluate a non-promoted ten-session swing candidate."""

        result = train_swing_edge_candidate(
            panel_authority_directory=panel_dir,
            output_directory=out_dir,
            strategy_contract=load_strategy_contract(contract),
            config=load_swing_training_config(policy),
            temporal_policy_path=temporal_policy,
        )
        console.print(
            {
                "status": result.evaluation["status"],
                "selected_candidate_id": result.selected_candidate_id,
                "final_test": result.evaluation.get("final_test"),
                "out_dir": str(result.output_directory),
            }
        )

    @app.command("plan-edge-rebuild-intraday-history")
    @serialized_heavy_job("plan-edge-rebuild-intraday-history")
    def plan_edge_rebuild_intraday_history(
        readiness_audit_dir: Path = typer.Option(...),
        memberships: Path = typer.Option(...),
        membership_audit: Path = typer.Option(...),
        existing_stock_bars_dir: Path = typer.Option(...),
        existing_benchmark_bars_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        policy: Path = typer.Option(Path("configs/edge_rebuild_intraday_history.toml")),
    ) -> None:
        """Plan causal PIT five-minute history before selective minute labels."""

        result = build_intraday_history_plan(
            readiness_audit_directory=readiness_audit_dir,
            memberships_path=memberships,
            membership_audit_path=membership_audit,
            existing_stock_bars_directory=existing_stock_bars_dir,
            existing_benchmark_bars_directory=(existing_benchmark_bars_dir),
            policy_path=policy,
            output_directory=out_dir,
            config=load_intraday_history_config(policy),
        )
        console.print(result["summary"])

    @app.command("screen-edge-rebuild-intraday-universe")
    @serialized_heavy_job("screen-edge-rebuild-intraday-universe")
    def screen_edge_rebuild_intraday_universe(
        canonical_dir: Path = typer.Option(
            ...,
            help="Verified merged canonical regular-session five-minute corpus.",
        ),
        out_dir: Path = typer.Option(..., help="Hash-bound research output directory."),
        first_session: str = typer.Option(..., help="Window start YYYY-MM-DD."),
        last_session: str = typer.Option(..., help="Window end YYYY-MM-DD."),
        contract: Path = typer.Option(
            Path("configs/edge_rebuild_strategy_contract.toml"),
            help="Frozen strategy contract supplying every screen threshold.",
        ),
        exclude_ticker: list[str] = typer.Option(
            [],
            help="Symbols collected as benchmarks rather than as candidates.",
        ),
    ) -> None:
        """Screen causal five-minute activity into selected stock-sessions."""

        result = build_intraday_selection(
            canonical_dir=canonical_dir,
            contract=load_strategy_contract(contract),
            first_session=date.fromisoformat(first_session),
            last_session=date.fromisoformat(last_session),
            exclude_tickers=frozenset(value.strip().upper() for value in exclude_ticker if value.strip()),
        )
        manifest = publish_intraday_selection(result, output_directory=out_dir)
        console.print(
            {
                key: manifest[key]
                for key in (
                    "symbols_read",
                    "five_minute_rows_read",
                    "activity_rows",
                    "sessions_in_window",
                    "layer_one",
                    "layer_two",
                )
            }
        )

    @app.command("audit-edge-rebuild-readiness")
    @serialized_heavy_job("audit-edge-rebuild-readiness")
    def audit_edge_rebuild_readiness(
        swing_panel_dir: Path = typer.Option(...),
        swing_candidate_dir: Path = typer.Option(...),
        swing_promoted_bundle_dir: Path | None = typer.Option(None),
        intraday_training_dir: Path = typer.Option(...),
        intraday_collection_dir: Path = typer.Option(...),
        intraday_coverage_dir: Path = typer.Option(...),
        catalyst_lineage_dir: Path = typer.Option(...),
        news_source_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        policy: Path = typer.Option(Path("configs/edge_rebuild_readiness.toml")),
        swing_training_policy: Path = typer.Option(Path("configs/edge_rebuild_swing_training.toml")),
        strategy_contract: Path = typer.Option(Path("configs/edge_rebuild_strategy_contract.toml")),
        intraday_policy: Path = typer.Option(Path("configs/intraday_specialist_research.toml")),
    ) -> None:
        """Audit independent source capacity without fitting a model."""

        result = run_edge_rebuild_readiness_audit(
            swing_panel_dir=swing_panel_dir,
            swing_candidate_dir=swing_candidate_dir,
            swing_promoted_bundle_dir=swing_promoted_bundle_dir,
            intraday_training_dir=intraday_training_dir,
            intraday_collection_dir=intraday_collection_dir,
            intraday_coverage_dir=intraday_coverage_dir,
            catalyst_lineage_dir=catalyst_lineage_dir,
            news_source_dir=news_source_dir,
            out_dir=out_dir,
            config=load_edge_rebuild_readiness_config(policy),
            policy_path=policy,
            swing_training_policy_path=swing_training_policy,
            strategy_contract_path=strategy_contract,
            intraday_config=load_intraday_specialist_research_config(intraday_policy),
            intraday_policy_path=intraday_policy,
        )
        console.print(result)
