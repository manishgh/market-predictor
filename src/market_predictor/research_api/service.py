"""Historical queries over exact saved temporal validation predictions.

No estimator is loaded or refitted. Final refits cannot supply OOF predictions.
These inspected development periods are not an untouched final test.
"""
from __future__ import annotations

import io
import math
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.research_api.catalog import load_research_model_specs, load_run, load_unit, model_state_record, verified_bytes


class ResearchModelUnavailableError(RuntimeError):
    pass


class ResearchFeatureUnavailableError(RuntimeError):
    pass


class ResearchModelService:
    def __init__(self, repository_root: Path, *, catalog_path: Path | None = None) -> None:
        self.repository_root = repository_root.resolve()
        self.specs = load_research_model_specs(catalog_path or Path(__file__).resolve().parents[3] / "configs/research_model_catalog.toml",
            repository_root=self.repository_root)

    def model_states(self) -> list[dict[str, Any]]:
        return [model_state_record(spec) for spec in self.specs.values()]

    def predict(self, *, model_id: str, tickers: list[str], session_date: date) -> dict[str, Any]:
        spec = self.specs[model_id]
        try:
            manifest, request = load_run(spec)
            folds = [fold for fold in request["folds"] if session_date.isoformat() in fold["score"]]
            if len(folds) != 1:
                raise ResearchFeatureUnavailableError("No saved temporal validation predictions exist for this exact session.")
            key = f"{spec.family}/temporal/fold-{folds[0]['number']}"
            directory, unit_manifest = load_unit(spec, manifest, request, key)
            unit = unit_manifest["unit"]
            cutoff = pd.Timestamp(unit["fit_cutoff_utc"])
            maturity = pd.Timestamp(unit["maximum_training_label_maturity"])
            if cutoff.tzinfo is None or maturity.tzinfo is None or maturity >= cutoff:
                raise DataReadinessError("temporal training label maturity is invalid")
            content = verified_bytes(directory / "predictions.parquet", unit_manifest["files"]["predictions.parquet"])
            columns = ["decision_id", "security_id", "ticker", "session_date_et", "decision_time_utc",
                "scope_eligible", "predicted_excess_return", "prediction_status"]
            frame = pd.read_parquet(io.BytesIO(content), columns=columns,
                filters=[("session_date_et", "==", session_date), ("ticker", "in", tickers)])
            if (frame.ticker.duplicated().any() or frame.decision_id.duplicated().any()
                    or not frame.session_date_et.eq(session_date).all() or not frame.ticker.isin(tickers).all()):
                raise DataReadinessError("saved prediction identities or sessions are ambiguous")
            predictions = []
            for row in frame.to_dict("records"):
                decision_time = pd.Timestamp(row["decision_time_utc"])
                if pd.isna(decision_time) or decision_time.tzinfo is None or decision_time < cutoff:
                    raise DataReadinessError("saved prediction precedes its temporal fit")
                eligible = row["scope_eligible"]
                value = row["predicted_excess_return"]
                available = not pd.isna(value)
                if (type(eligible) is not bool or eligible != available
                        or row["prediction_status"] != ("research_prediction" if eligible else "outside_eligible_scope")
                        or (available and not math.isfinite(float(value)))):
                    raise DataReadinessError("saved return value or eligibility is inconsistent")
                predictions.append({"ticker": row["ticker"], "security_id": row["security_id"],
                    "decision_id": row["decision_id"], "decision_time_utc": decision_time.isoformat(),
                    "predicted_excess_return": float(value) if available else None,
                    "prediction_status": row["prediction_status"], "actionable": False})
            present = set(frame.ticker)
            return {"model": {"id": spec.model_id, "label": spec.label, "family": spec.family},
                "session_date": session_date.isoformat(), "target": request["policy"]["target"],
                "horizon_sessions": 10, "return_unit": "fraction", "prediction_basis": "saved_temporal_out_of_fold",
                "assessment_status": "inspected_development_evidence", "actionable": False,
                "predictions": sorted(predictions, key=lambda row: tickers.index(row["ticker"])),
                "rejected": [{"ticker": ticker, "reason": "No saved row for this exact session."}
                    for ticker in tickers if ticker not in present],
                "provenance": {"run_manifest_sha256": spec.manifest_sha256,
                    "request_sha256": manifest["request_sha256"], "unit": key,
                    "unit_manifest_sha256": manifest["units"][key]["manifest_sha256"],
                    "model_sha256": unit_manifest["files"]["model.joblib"],
                    "predictions_sha256": unit_manifest["files"]["predictions.parquet"],
                    "fit_cutoff_utc": cutoff.isoformat()}}
        except (OSError, ValueError, KeyError, TypeError, DataReadinessError) as exc:
            raise ResearchModelUnavailableError("Pinned model or prediction artifacts are missing, modified or inconsistent.") from exc
