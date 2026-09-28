from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import typer

from market_predictor.canonical.store import file_sha256
from market_predictor.config import get_settings
from market_predictor.core import path_integrity
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.governance.outcomes.repository import OutcomeRepository
from market_predictor.governance.outcomes.session_records import SessionRecordStore, make_session_record
from market_predictor.monitoring_lease import monitoring_lease
from market_predictor.serving.live_input_publication import LiveInputPublication, publish_live_inputs
from market_predictor.serving.prediction_service import (
    PredictionService,
    serving_routes_from_config,
    swing_live_input_provider_from_config,
)
from market_predictor.serving.session_registration import register_session_predictions


def _service() -> PredictionService:
    settings = get_settings()
    return PredictionService(
        Path("."), routes=serving_routes_from_config(settings.app_config),
        swing_live_input_provider=swing_live_input_provider_from_config(
            settings.app_config, memory_budget_gib=settings.runtime_memory_budget_gib,
            memory_headroom_gib=settings.runtime_memory_headroom_gib,
        ),
        memory_budget_gib=settings.runtime_memory_budget_gib, memory_headroom_gib=settings.runtime_memory_headroom_gib,
        max_concurrent_inference=settings.runtime_max_concurrent_inference,
        max_tickers_per_request=settings.runtime_max_tickers_per_request,
        inference_memory_reservation_gib=settings.runtime_inference_memory_reservation_gib,
        reject_unknown_memory=settings.runtime_reject_unknown_memory,
    )


def register_session_monitoring_commands(app: typer.Typer, console: Any) -> None:
    @app.command("publish-swing-live-inputs")
    def publish_inputs(
        request_path: Path = typer.Option(..., help="Strict publication JSON with absolute paths and independent source pins."),
        expected_request_sha256: str = typer.Option(...),
        output_directory: Path = typer.Option(Path("data/live/edge_rebuild/swing")),
        maximum_bytes: int = typer.Option(512 * 1024 * 1024, min=1),
        maximum_rows: int = typer.Option(2_000_000, min=1),
    ) -> None:
        """Verify prepared causal inputs and activate one immutable nightly generation."""
        safe = path_integrity.verify_no_reparse_ancestry(request_path, label="live publication request")
        if safe.stat().st_size > 1_048_576 or file_sha256(safe) != expected_request_sha256:
            raise typer.BadParameter("publication request hash or byte limit does not verify")
        request = LiveInputPublication.model_validate(parse_strict_json_object(safe.read_bytes(), label="live publication request"))
        if file_sha256(safe) != expected_request_sha256:
            raise typer.BadParameter("publication request changed during loading")
        result = publish_live_inputs(request, output_directory, maximum_bytes=maximum_bytes, maximum_rows=maximum_rows)
        console.print_json(data=result)

    @app.command("register-session-predictions")
    def register_session(
        as_of: datetime = typer.Option(..., formats=["%Y-%m-%dT%H:%M:%S%z"]),
        outcome_dir: Path = typer.Option(Path("data/predictions/outcomes")),
    ) -> None:
        """Register a full nightly cross-section, including every abstaining member."""
        if as_of.utcoffset() is None:
            raise typer.BadParameter("--as-of requires an explicit timezone offset")
        result = register_session_predictions(_service(), OutcomeRepository(outcome_dir), as_of=as_of)
        console.print_json(result.model_dump_json())

    @app.command("record-monitoring-not-run")
    def record_not_run(
        route_key: str = typer.Option(...),
        session: str = typer.Option(..., help="Missed XNYS session, YYYY-MM-DD."),
        operator_id: str = typer.Option(...),
        reason: str = typer.Option(...),
        outcome_dir: Path = typer.Option(Path("data/predictions/outcomes")),
    ) -> None:
        """Audit a missed session against a previously verified route activation."""
        store = SessionRecordStore(outcome_dir)
        with monitoring_lease("record-monitoring-not-run"):
            route = store.load_route(route_key)
            result = store.record(make_session_record(
                route=route, decision_session=date.fromisoformat(session), status="failed", failure_reason="not_run",
                recorded_at_utc=datetime.now(UTC), operator_id=operator_id.strip(), operator_reason=reason.strip(),
            ))
        console.print_json(result.model_dump_json())
