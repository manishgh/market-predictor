from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, cast

import joblib
import numpy as np
import pandas as pd

from market_predictor.feature_store import LiveFeatureStore
from market_predictor.research_api.catalog import (
    ResearchModelState,
    inspect_research_model,
    load_research_model_specs,
)


class ResearchModelUnavailableError(RuntimeError):
    pass


class ResearchFeatureUnavailableError(RuntimeError):
    pass


class ResearchFeatureSource(Protocol):
    def load(self, mode: str, *, as_of: datetime) -> pd.DataFrame: ...


class RegisteredResearchFeatureSource:
    """Read integrity-checked snapshots built by the canonical feature pipeline."""

    def __init__(self, repository_root: Path) -> None:
        self._store = LiveFeatureStore(repository_root)

    def load(self, mode: str, *, as_of: datetime) -> pd.DataFrame:
        if mode not in {"swing", "intraday"}:
            raise ValueError(f"unsupported research feature mode: {mode}")
        try:
            return self._store.load(
                cast(Any, mode),
                as_of=as_of,
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise ResearchFeatureUnavailableError(str(exc)) from exc


class ResearchModelService:
    """Non-actionable scoring over canonical, registered live feature snapshots."""

    def __init__(
        self,
        repository_root: Path,
        *,
        catalog_path: Path | None = None,
        feature_source: ResearchFeatureSource | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.catalog_path = catalog_path or (
            self.repository_root / "configs/research_model_catalog.toml"
        )
        self.specs = load_research_model_specs(
            self.catalog_path,
            repository_root=self.repository_root,
        )
        self.feature_source = feature_source or RegisteredResearchFeatureSource(
            self.repository_root
        )
        self._payload_cache: dict[str, tuple[str, Mapping[str, Any]]] = {}
        self._cache_lock = Lock()

    def model_states(self) -> list[ResearchModelState]:
        return [inspect_research_model(self.specs[key]) for key in sorted(self.specs)]

    def model_state(self, model_id: str) -> ResearchModelState:
        spec = self.specs.get(model_id)
        if spec is None:
            raise KeyError(model_id)
        return inspect_research_model(spec)

    def predict(
        self,
        *,
        model_id: str,
        tickers: list[str],
        as_of: datetime | None = None,
    ) -> dict[str, Any]:
        state = self.model_state(model_id)
        if not state.research_scoring_available or state.artifact_sha256 is None:
            raise ResearchModelUnavailableError(state.reason)
        cutoff = as_of or datetime.now(UTC)
        if cutoff.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        frame = self.feature_source.load(state.spec.mode, as_of=cutoff.astimezone(UTC))
        selected, rejected = _select_requested_rows(frame, tickers)
        if selected.empty:
            return {
                "model": _state_record(state),
                "as_of_utc": cutoff.astimezone(UTC).isoformat(),
                "predictions": [],
                "rejected": rejected,
                "actionable": False,
            }

        payload = self._load_payload(state)
        expected_columns = _expected_columns(payload)
        missing = sorted(set(expected_columns).difference(selected.columns))
        if missing:
            raise ResearchFeatureUnavailableError(
                "registered live feature snapshot does not match the model contract; "
                f"missing columns: {missing[:12]}"
            )
        matrix = selected.loc[:, expected_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        values = matrix.to_numpy(dtype="float64")
        if not bool(np.isfinite(values).all()):
            raise ResearchFeatureUnavailableError(
                "registered live feature snapshot contains unavailable model inputs"
            )

        primary_scores, auxiliary_scores = _score_payload(payload, matrix)
        predictions: list[dict[str, Any]] = []
        for position, (_, row) in enumerate(selected.iterrows()):
            evidence = {
                column: float(matrix.iloc[position][column])
                for column in expected_columns
            }
            auxiliary = {
                key: float(scores[position])
                for key, scores in auxiliary_scores.items()
            }
            predictions.append(
                {
                    "ticker": str(row["ticker"]),
                    "score": float(primary_scores[position]),
                    "auxiliary_scores": auxiliary,
                    "feature_available_at_utc": _optional_iso(
                        row.get("feature_available_at_utc")
                    ),
                    "features": evidence,
                    "actionable": False,
                }
            )
        predictions.sort(key=lambda item: item["score"], reverse=True)
        for rank, prediction in enumerate(predictions, start=1):
            prediction["rank"] = rank
        return {
            "model": _state_record(state),
            "as_of_utc": cutoff.astimezone(UTC).isoformat(),
            "predictions": predictions,
            "rejected": rejected,
            "actionable": False,
        }

    def _load_payload(self, state: ResearchModelState) -> Mapping[str, Any]:
        assert state.artifact_sha256 is not None
        cached = self._payload_cache.get(state.spec.model_id)
        if cached is not None and cached[0] == state.artifact_sha256:
            return cached[1]
        with self._cache_lock:
            cached = self._payload_cache.get(state.spec.model_id)
            if cached is not None and cached[0] == state.artifact_sha256:
                return cached[1]
            loaded = joblib.load(state.spec.artifact_directory / "candidate.joblib")
            if not isinstance(loaded, Mapping):
                raise ResearchModelUnavailableError(
                    "candidate payload does not contain a model mapping"
                )
            payload = {str(key): value for key, value in loaded.items()}
            self._payload_cache[state.spec.model_id] = (
                state.artifact_sha256,
                payload,
            )
            return payload


def _select_requested_rows(
    frame: pd.DataFrame,
    tickers: list[str],
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    if "ticker" not in frame.columns:
        raise ResearchFeatureUnavailableError(
            "registered live feature snapshot is missing ticker identity"
        )
    normalized = frame.copy()
    normalized["ticker"] = normalized["ticker"].astype(str).str.upper().str.strip()
    if bool(normalized["ticker"].duplicated().any()):
        raise ResearchFeatureUnavailableError(
            "registered live feature snapshot contains duplicate ticker rows"
        )
    indexed = normalized.set_index("ticker", drop=False)
    present = [ticker for ticker in tickers if ticker in indexed.index]
    rejected = [
        {"ticker": ticker, "reason": "No current registered feature row is available."}
        for ticker in tickers
        if ticker not in indexed.index
    ]
    return indexed.loc[present].reset_index(drop=True), rejected


def _expected_columns(payload: Mapping[str, Any]) -> list[str]:
    raw = payload.get("feature_columns")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ResearchModelUnavailableError(
            "candidate payload is missing its ordered feature contract"
        )
    columns = [str(column) for column in raw]
    if len(columns) != len(set(columns)):
        raise ResearchModelUnavailableError(
            "candidate payload contains duplicate feature columns"
        )
    return columns


def _score_payload(
    payload: Mapping[str, Any],
    matrix: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    fitted_models = payload.get("fitted_models")
    if isinstance(fitted_models, Mapping) and fitted_models:
        auxiliary = {
            str(name): _score_estimator(model, matrix)
            for name, model in fitted_models.items()
        }
        primary_name = (
            "xgboost_regressor"
            if "xgboost_regressor" in auxiliary
            else "classifier"
            if "classifier" in auxiliary
            else next(iter(auxiliary))
        )
        return auxiliary[primary_name], auxiliary
    fitted_candidate = payload.get("fitted_candidate")
    if fitted_candidate is None:
        raise ResearchModelUnavailableError(
            "candidate payload does not contain a fitted estimator"
        )
    scores = _score_estimator(fitted_candidate, matrix)
    return scores, {"candidate": scores}


def _score_estimator(model: Any, matrix: pd.DataFrame) -> np.ndarray:
    estimator = getattr(model, "estimator", model)
    model_columns = tuple(
        str(column) for column in getattr(model, "feature_columns", ())
    )
    if model_columns:
        missing = sorted(set(model_columns).difference(matrix.columns))
        if missing:
            raise ResearchModelUnavailableError(
                f"candidate feature subset is unavailable: {missing}"
            )
        model_matrix = matrix.loc[:, model_columns]
    else:
        model_matrix = matrix
    predict_proba = getattr(estimator, "predict_proba", None)
    if callable(predict_proba):
        probabilities = np.asarray(predict_proba(model_matrix), dtype="float64")
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ResearchModelUnavailableError(
                "classifier returned an invalid probability matrix"
            )
        result = probabilities[:, 1]
    else:
        predict = getattr(estimator, "predict", None)
        if not callable(predict):
            raise ResearchModelUnavailableError(
                "candidate estimator does not expose a prediction method"
            )
        result = np.asarray(predict(model_matrix), dtype="float64").reshape(-1)
    if len(result) != len(matrix) or not bool(np.isfinite(result).all()):
        raise ResearchModelUnavailableError(
            "candidate estimator returned invalid scores"
        )
    return result


def _state_record(state: ResearchModelState) -> dict[str, Any]:
    return {
        "id": state.spec.model_id,
        "label": state.spec.label,
        "mode": state.spec.mode,
        "uses_catalyst": state.spec.uses_catalyst,
        "training_status": state.training_status,
        "artifact_available": state.artifact_available,
        "integrity_verified": state.integrity_verified,
        "research_scoring_available": state.research_scoring_available,
        "promotion_permitted": state.promotion_permitted,
        "candidate_id": state.candidate_id,
        "reason": state.reason,
        "artifact_sha256": state.artifact_sha256,
    }


def model_state_record(state: ResearchModelState) -> dict[str, Any]:
    return _state_record(state)


def _optional_iso(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ResearchFeatureUnavailableError(
            "feature availability timestamp is not timezone-aware"
        )
    return str(timestamp.tz_convert("UTC").isoformat())
