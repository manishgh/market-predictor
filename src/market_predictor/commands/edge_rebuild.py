"""Edge-rebuild research audit commands."""

from datetime import date
from pathlib import Path
from typing import Any

import typer

from market_predictor.config import get_settings
from market_predictor.edge_rebuild.contracts import (
    load_edge_rebuild_readiness_config,
)
from market_predictor.edge_rebuild.extended_session_context import (
    build_extended_session_context_plan,
)
from market_predictor.edge_rebuild.history_collection import (
    collect_intraday_history,
)
from market_predictor.edge_rebuild.history_contracts import (
    load_collection_transport_config,
    load_extended_session_context_config,
    load_intraday_history_config,
    load_selected_session_history_config,
)
from market_predictor.edge_rebuild.history_materialization import (
    reorganize_intraday_history,
)
from market_predictor.edge_rebuild.intraday_history import (
    build_intraday_history_plan,
)
from market_predictor.edge_rebuild.intraday_selection import (
    build_intraday_selection,
    publish_intraday_selection,
)
from market_predictor.edge_rebuild.readiness import (
    run_edge_rebuild_readiness_audit,
)
from market_predictor.edge_rebuild.selected_session_history import (
    build_selected_session_history_plan,
)
from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract
from market_predictor.edge_rebuild.swing_history_acquisition import (
    publish_swing_history_acquisition_plan,
)
from market_predictor.edge_rebuild.swing_materialization import (
    materialize_swing_feature_panel,
)
from market_predictor.edge_rebuild.swing_ordering import audit_swing_ordering
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
from market_predictor.v3.errors import DataReadinessError


def register_edge_rebuild_commands(app: typer.Typer, console: Any) -> None:
    @app.command("plan-edge-rebuild-swing-history")
    @serialized_heavy_job("plan-edge-rebuild-swing-history")
    def plan_edge_rebuild_swing_history(
        temporal_manifest_dir: Path = typer.Option(...),
        memberships: Path = typer.Option(...),
        universe_audit: Path = typer.Option(...),
        current_daily_collection_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        repository_root: Path = typer.Option(Path(".")),
    ) -> None:
        """Plan membership-first acquisition for missing swing history."""

        result = publish_swing_history_acquisition_plan(
            repository_root=repository_root,
            temporal_manifest_directory=temporal_manifest_dir,
            memberships_path=memberships,
            universe_audit_path=universe_audit,
            current_daily_collection_directory=current_daily_collection_dir,
            output_directory=out_dir,
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
                        "required_start",
                        "required_end",
                        "reusable_official_sources",
                        "total_official_sources",
                        "invalid_official_sources",
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
        policy: Path = typer.Option(
            Path("configs/edge_rebuild_temporal_manifest.toml")
        ),
        contract: Path = typer.Option(
            Path("configs/edge_rebuild_strategy_contract.toml")
        ),
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
        collection_dir: Path = typer.Option(
            ...,
            help="Completed canonical daily-bar collection.",
        ),
        memberships: Path = typer.Option(
            ...,
            help="Verified point-in-time membership artifact.",
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
    ) -> None:
        """Publish the complete seven-year causal swing ranking panel."""

        result = materialize_swing_feature_panel(
            collection_dir=collection_dir,
            memberships_path=memberships,
            contract=load_strategy_contract(contract),
            output_dir=out_dir,
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
            frame["session"] = (
                frame["bar_start_utc"]
                .dt.tz_convert("America/New_York")
                .dt.date.astype(str)
            )
            frames.append(
                frame.rename(columns={"close": "last_close"}).assign(bars=1)[
                    ["session", "ticker", "bars", "last_close"]
                ]
            )
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
            help=(
                "Published two-layer screen making those stock-sessions "
                "eligible; required with --selected-session-collection-dir."
            ),
        ),
    ) -> None:
        """Reorganize downloaded bars into per-symbol regular and extended stores."""

        if (selected_session_collection_dir is None) != (selected_sessions is None):
            raise DataReadinessError(
                "selected-session bars and their published screen must be "
                "supplied together"
            )
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
        policy: Path = typer.Option(
            Path("configs/edge_rebuild_extended_session_context.toml")
        ),
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
            help=(
                "Collection policy for the layer being collected. The layer "
                "comes from the policy's own schema_version."
            ),
        ),
    ) -> None:
        """Collect resumable PIT SIP five-minute bars for any planned layer."""

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
        policy: Path = typer.Option(
            Path("configs/edge_rebuild_selected_session_history.toml")
        ),
    ) -> None:
        """Plan five-minute bars for exactly the selected in-play stock-sessions."""

        result = build_selected_session_history_plan(
            selection_directory=selection_dir,
            policy_path=policy,
            output_directory=out_dir,
            config=load_selected_session_history_config(policy),
        )
        console.print(result["summary"])

    @app.command("plan-edge-rebuild-intraday-history")
    @serialized_heavy_job("plan-edge-rebuild-intraday-history")
    def plan_edge_rebuild_intraday_history(
        readiness_audit_dir: Path = typer.Option(...),
        memberships: Path = typer.Option(...),
        membership_audit: Path = typer.Option(...),
        existing_stock_bars_dir: Path = typer.Option(...),
        existing_benchmark_bars_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        policy: Path = typer.Option(
            Path("configs/edge_rebuild_intraday_history.toml")
        ),
    ) -> None:
        """Plan causal PIT five-minute history before selective minute labels."""

        result = build_intraday_history_plan(
            readiness_audit_directory=readiness_audit_dir,
            memberships_path=memberships,
            membership_audit_path=membership_audit,
            existing_stock_bars_directory=existing_stock_bars_dir,
            existing_benchmark_bars_directory=(
                existing_benchmark_bars_dir
            ),
            policy_path=policy,
            output_directory=out_dir,
            config=load_intraday_history_config(policy),
        )
        console.print(result["summary"])

    @app.command("screen-edge-rebuild-intraday-universe")
    @serialized_heavy_job("screen-edge-rebuild-intraday-universe")
    def screen_edge_rebuild_intraday_universe(
        collection_dir: Path = typer.Option(
            ...,
            help="Completed daily-bar collection supplying volume and price.",
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
        """Screen daily bars into the frozen two-layer intraday stock-sessions."""

        result = build_intraday_selection(
            collection_dir=collection_dir,
            contract=load_strategy_contract(contract),
            first_session=date.fromisoformat(first_session),
            last_session=date.fromisoformat(last_session),
            exclude_tickers=frozenset(
                value.strip().upper() for value in exclude_ticker if value.strip()
            ),
        )
        manifest = publish_intraday_selection(result, output_directory=out_dir)
        console.print(
            {
                key: manifest[key]
                for key in (
                    "symbols_read",
                    "daily_rows_read",
                    "zero_volume_rows_dropped",
                    "liquidity_rows",
                    "sessions_in_window",
                    "layer_one",
                    "layer_two",
                )
            }
        )

    @app.command("audit-edge-rebuild-readiness")
    @serialized_heavy_job("audit-edge-rebuild-readiness")
    def audit_edge_rebuild_readiness(
        swing_bundle_dir: Path = typer.Option(...),
        swing_technical_path: Path = typer.Option(...),
        intraday_training_dir: Path = typer.Option(...),
        intraday_collection_dir: Path = typer.Option(...),
        intraday_coverage_dir: Path = typer.Option(...),
        catalyst_lineage_dir: Path = typer.Option(...),
        news_source_dir: Path = typer.Option(...),
        out_dir: Path = typer.Option(...),
        policy: Path = typer.Option(
            Path("configs/edge_rebuild_readiness.toml")
        ),
        swing_policy: Path = typer.Option(
            Path("configs/swing_specialist_research.toml")
        ),
        intraday_policy: Path = typer.Option(
            Path("configs/intraday_specialist_research.toml")
        ),
    ) -> None:
        """Audit independent source capacity without fitting a model."""

        result = run_edge_rebuild_readiness_audit(
            swing_bundle_dir=swing_bundle_dir,
            swing_technical_path=swing_technical_path,
            intraday_training_dir=intraday_training_dir,
            intraday_collection_dir=intraday_collection_dir,
            intraday_coverage_dir=intraday_coverage_dir,
            catalyst_lineage_dir=catalyst_lineage_dir,
            news_source_dir=news_source_dir,
            out_dir=out_dir,
            config=load_edge_rebuild_readiness_config(policy),
            policy_path=policy,
            swing_policy_path=swing_policy,
            intraday_config=load_intraday_specialist_research_config(
                intraday_policy
            ),
            intraday_policy_path=intraday_policy,
        )
        console.print(result)
