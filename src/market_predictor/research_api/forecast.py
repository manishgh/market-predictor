"""Fresh research forecasts using frozen regressors and registered causal inputs."""
from __future__ import annotations

import math
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from market_predictor.admission import InferenceAdmissionController
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.io import inside
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research_api.catalog import PROFILES, load_run, load_unit
from market_predictor.research_api.service import (
    ResearchFeatureUnavailableError,
    ResearchModelService,
    ResearchModelUnavailableError,
)
from market_predictor.resources import assert_memory_budget
from market_predictor.serving.swing_features import _expected_swing_decision_time
from market_predictor.swing.features.observed_return_publication import verify_observed_return_publication
from market_predictor.swing.training.return_artifacts import load_return_model
from market_predictor.swing.training.return_estimators import predict_return


class ReturnForecastConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    maximum_snapshot_age_seconds: int = Field(default=129_600, gt=0)
    maximum_snapshot_bytes: int = Field(default=67_108_864, gt=0)
    maximum_snapshot_rows: int = Field(default=2_000, gt=0)
    publication_directory: str = "data/live/returns/current"
    publication_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise DataReadinessError("forecast inputs require explicit UTC availability clocks")
    return stamp.tz_convert("UTC")


class ReturnForecastService:
    """Never rebuild, refit, or substitute a stored prediction for fresh inference."""

    def __init__(self, models: ResearchModelService, *, memory_budget_gib: float, memory_headroom_gib: float,
                 admission: InferenceAdmissionController | None = None,
                 inference_memory_reservation_gib: float = 0.5) -> None:
        self.models = models
        self.memory_budget_gib = memory_budget_gib
        self.memory_headroom_gib = memory_headroom_gib
        self.admission = admission or InferenceAdmissionController(max_concurrent_requests=1,
            memory_budget_gib=memory_budget_gib, memory_headroom_gib=memory_headroom_gib)
        if inference_memory_reservation_gib <= 0:
            raise ValueError("inference memory reservation must be positive")
        self.inference_memory_reservation_gib = inference_memory_reservation_gib
        with models.catalog_path.open("rb") as stream:
            self.config = ReturnForecastConfig.model_validate(tomllib.load(stream).get("forecast", {}))
        inside(models.repository_root, self.config.publication_directory)

    def publication_configured(self) -> bool:
        return self.config.publication_manifest_sha256 is not None

    def inputs_published(self) -> bool:
        directory = inside(self.models.repository_root, self.config.publication_directory)
        return self.publication_configured() and (directory / "_manifest.json").is_file()

    def forecast(self, *, model_id: str, tickers: list[str], as_of: datetime) -> dict[str, Any]:
        spec = self.models.specs[model_id]
        cutoff = _utc(as_of)
        if cutoff > pd.Timestamp(datetime.now(UTC)):
            raise ResearchFeatureUnavailableError("Future forecast cutoffs are unavailable.")
        try:
            runtime = heavy_job_runtime_dir()
            if not runtime.is_absolute():
                runtime = self.models.repository_root / runtime
            with (
                self.admission.lease(estimated_incremental_gib=self.inference_memory_reservation_gib),
                heavy_job_lease("return-model-forecast", runtime_dir=runtime),
            ):
                guard("return forecast")
                assert_memory_budget(hard_budget_gib=self.memory_budget_gib,
                    headroom_gib=self.memory_headroom_gib, stage="return forecast")
                manifest, request = load_run(spec)
                key = f"{spec.family}/final_refit"
                directory, unit = load_unit(spec, manifest, request, key)
                frame, provenance = self._features(spec.profile, spec.manifest_sha256,
                    request["feature_names"], cutoff)
                fit_cutoff = _utc(unit["unit"]["fit_cutoff_utc"])
                if _utc(frame.decision_time_utc.iloc[0]) <= fit_cutoff:
                    raise ResearchFeatureUnavailableError("Fresh forecasts must follow the model's final training cutoff.")
                selected = frame.loc[frame.ticker.isin(tickers)].copy()
                eligible = selected.loc[selected.forecast_eligible]
                by_decision: dict[str, float] = {}
                if not eligible.empty:
                    model = load_return_model(directory,
                        manifest_sha256=manifest["units"][key]["manifest_sha256"], purpose="research")
                    predictions = predict_return(model, eligible.loc[:, request["feature_names"]])
                    by_decision = {str(identity): float(value)
                        for identity, value in zip(eligible.decision_id, predictions, strict=True)}
                values = [{"ticker": row.ticker, "security_id": row.security_id,
                    "decision_id": row.decision_id, "decision_time_utc": _utc(row.decision_time_utc).isoformat(),
                    "predicted_excess_return": by_decision.get(row.decision_id),
                    "prediction_status": "research_forecast" if row.forecast_eligible else "unavailable",
                    "abstention_reason": None if row.forecast_eligible else row.forecast_abstention_reason,
                    "actionable": False} for row in selected.itertuples()]
                assert_memory_budget(hard_budget_gib=self.memory_budget_gib,
                    headroom_gib=self.memory_headroom_gib, stage="after return forecast")
                # Detect a publisher replacing an input during inference.
                if file_sha256(Path(provenance["feature_path"])) != provenance["feature_sha256"]:
                    raise ResearchFeatureUnavailableError("Forecast inputs changed during inference.")
                return {"model": {"id": spec.model_id, "label": spec.label, "family": spec.family},
                    "as_of_utc": cutoff.isoformat(), "session_date": str(frame.session_date_et.iloc[0]),
                    "target": request["policy"]["target"], "horizon_sessions": 10, "return_unit": "fraction",
                    "prediction_basis": "frozen_final_model_fresh_inputs", "boundary": "research_only",
                    "serving_eligible": False, "promotion_eligible": False, "actionable": False,
                    "predictions": sorted(values, key=lambda row: tickers.index(row["ticker"])),
                    "rejected": [{"ticker": ticker, "reason": "Ticker is absent from the current verified feature population."}
                        for ticker in tickers if ticker not in set(selected.ticker)],
                    "provenance": {**{name: value for name, value in provenance.items() if name != "feature_path"},
                        "run_manifest_sha256": spec.manifest_sha256, "request_sha256": manifest["request_sha256"],
                        "unit": key, "model_sha256": unit["files"]["model.joblib"], "fit_cutoff_utc": fit_cutoff.isoformat()}}
        except ResearchFeatureUnavailableError:
            raise
        except (HeavyJobBusyError, MemoryError, MemoryBudgetError) as exc:
            raise ResearchFeatureUnavailableError("Forecast resources are currently unavailable.") from exc
        except (OSError, ValueError, KeyError, TypeError, DataReadinessError) as exc:
            raise ResearchModelUnavailableError("Pinned model or fresh-input artifacts are missing, modified or inconsistent.") from exc

    def _features(self, profile: str, run_sha: str,
                  names: list[str], cutoff: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
        digest = self.config.publication_manifest_sha256
        if digest is None:
            raise ResearchFeatureUnavailableError("Current observed stock/news features have not been published and registered.")
        directory = inside(self.models.repository_root, self.config.publication_directory)
        try:
            frame, provenance = verify_observed_return_publication(
                root=self.models.repository_root, directory=directory,
                expected_manifest_sha256=digest, as_of=cutoff, profile=profile,
                feature_names=names, run_manifest_sha256=run_sha,
                maximum_bytes=self.config.maximum_snapshot_bytes,
                maximum_rows=self.config.maximum_snapshot_rows)
        except (OSError, ValueError, KeyError, TypeError, DataReadinessError) as exc:
            raise ResearchFeatureUnavailableError("Registered observed feature publication is unavailable or inconsistent.") from exc
        required = {"ticker", "security_id", "decision_id", "session_date_et", "decision_time_utc",
                    "feature_available_at_utc", "availability_semantics", "feature_profile",
                    "forecast_eligible", "forecast_abstention_reason", *names}
        if (frame.empty or len(frame) > self.config.maximum_snapshot_rows or not frame.columns.is_unique
                or not required.issubset(frame) or frame.ticker.duplicated().any()
                or frame.security_id.duplicated().any() or frame.decision_id.duplicated().any()
                or frame.forecast_eligible.isna().any() or frame.forecast_eligible.dtype.kind != "b"
                or not frame.loc[~frame.forecast_eligible, "forecast_abstention_reason"].map(
                    lambda value: isinstance(value, str) and bool(value)).all()
                or not frame.feature_profile.eq(PROFILES[profile]).all()
                or not frame.availability_semantics.eq("observed").all()):
            raise ResearchFeatureUnavailableError("Fresh feature identities, population or observed source clocks are invalid.")
        created = _utc(provenance["created_at_utc"])
        expected = _expected_swing_decision_time(cutoff)
        decisions = frame.decision_time_utc.map(_utc)
        available = frame.feature_available_at_utc.map(_utc)
        if (created > cutoff or (cutoff - created).total_seconds() > self.config.maximum_snapshot_age_seconds
                or expected > cutoff or not decisions.eq(expected).all()
                or not available.le(decisions).all() or not available.le(cutoff).all()
                or not pd.to_datetime(frame.session_date_et).dt.date.eq(expected.tz_convert("America/New_York").date()).all()):
            raise ResearchFeatureUnavailableError("Fresh feature publication is stale, future-dated or for a different decision session.")
        numeric = frame.loc[:, names]
        if any(getattr(numeric[name].dtype, "kind", "") not in "iuf" for name in names):
            raise ResearchFeatureUnavailableError("Fresh model features must use the frozen numerical feature contract.")
        if any(not (pd.isna(value) or math.isfinite(float(value))) for value in numeric.to_numpy().flat):
            raise ResearchFeatureUnavailableError("Fresh features contain invalid numerical values.")
        return frame, provenance
