from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import typer
from rich.console import Console

from market_predictor.registry import file_sha256
from market_predictor.v3.evaluation import (
    RankingAuditConfig,
    build_multi_output_evidence,
    evaluate_ranking_economics,
    fit_disjoint_calibrator,
)


def register_v3_evaluation_commands(app: typer.Typer, console: Console) -> None:
    @app.command("audit-v3-ranking")
    def audit_v3_ranking(
        predictions: Path = typer.Option(..., help="Combined V3 OOF prediction parquet."),
        opportunity_family: str = typer.Option("R1", help="Opportunity family to rank."),
        downside_family: str = typer.Option("D1", help="Probabilistic downside family."),
        calibration_method: str = typer.Option("sigmoid", help="Downside calibration: sigmoid or isotonic."),
        top_k: int = typer.Option(10, min=1, help="Maximum selected names per decision group."),
        maximum_downside_probability: float = typer.Option(0.5, min=0, max=1, help="Calibrated downside veto threshold."),
        bootstrap_iterations: int = typer.Option(1_000, min=100, help="Whole-session bootstrap iterations."),
        minimum_sessions: int = typer.Option(20, min=2, help="Minimum independent evaluation sessions."),
        report_out: Path = typer.Option(Path("data/reports/v3_ranking_audit_latest.json"), help="Combined audit report JSON."),
        selected_out: Path = typer.Option(Path("data/reports/v3_selected_oof_latest.parquet"), help="Independent selected-trade parquet."),
        calibration_out: Path = typer.Option(
            Path("models/v3/calibration/d1_audit_calibrator.joblib"),
            help="Audit-only calibrator artifact.",
        ),
        overwrite: bool = typer.Option(False, help="Explicitly replace audit artifacts."),
    ) -> None:
        """Calibrate D1 on disjoint OOF sessions and audit top-k ranking economics."""
        if calibration_method not in {"sigmoid", "isotonic"}:
            raise typer.BadParameter("calibration-method must be sigmoid or isotonic")
        for path in (report_out, selected_out, calibration_out):
            if path.exists() and not overwrite:
                raise typer.BadParameter(f"Output already exists; pass --overwrite to replace it: {path}")
        frame = _read_predictions(predictions)
        calibrator, calibration_report, calibration_evaluation = fit_disjoint_calibrator(
            frame,
            family=downside_family,
            method=calibration_method,  # type: ignore[arg-type]
            minimum_sessions=max(6, minimum_sessions * 2),
        )
        evidence = build_multi_output_evidence(
            frame,
            opportunity_family=opportunity_family,
            downside_family=downside_family,
            downside_calibration=calibration_evaluation,
        )
        ranking_report, selected = evaluate_ranking_economics(
            evidence,
            config=RankingAuditConfig(
                top_k=top_k,
                maximum_downside_probability=maximum_downside_probability,
                bootstrap_iterations=bootstrap_iterations,
                minimum_sessions=minimum_sessions,
            ),
        )
        report: dict[str, object] = {
            "schema": "ml_v3.combined_evidence_audit.v1",
            "calibration": calibration_report,
            "ranking_economics": ranking_report,
            "promotion_gates": "not_evaluated_until_c8_gate_freeze",
        }
        report_out.parent.mkdir(parents=True, exist_ok=True)
        selected_out.parent.mkdir(parents=True, exist_ok=True)
        calibration_out.parent.mkdir(parents=True, exist_ok=True)
        selected.to_parquet(selected_out, index=False)
        joblib.dump(
            {
                "schema": "ml_v3.audit_calibrator.v1",
                "calibrator": calibrator,
                "report": calibration_report,
                "serving_eligible": False,
            },
            calibration_out,
        )
        report["artifacts"] = {
            "selected_predictions": {"path": str(selected_out), "sha256": file_sha256(selected_out)},
            "audit_calibrator": {
                "path": str(calibration_out),
                "sha256": file_sha256(calibration_out),
                "serving_eligible": False,
            },
        }
        report_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        console.print(f"Wrote V3 ranking audit to {report_out}")
        console.print(f"Wrote {len(selected)} independent selected rows to {selected_out}")


def _read_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Missing predictions: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise typer.BadParameter(f"Unsupported predictions format: {path.suffix}")
