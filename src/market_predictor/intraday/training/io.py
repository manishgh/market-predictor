"""Development-only, cost-aware intraday model training and evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, cast

import joblib
import numpy as np
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.intraday.training.config import BaselineProfile, IntradayDevelopmentConfig, _CandidateSpec
from market_predictor.intraday.training.event_training import (
    DIRECTIONAL_EVENT_SUBTYPES,
)
from market_predictor.intraday.training.training import (
    MODEL_FEATURE_COLUMNS,
    PublishedIntradayDataset,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
)

MODEL_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_candidate.v1"
EVALUATION_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_evaluation.v1"
AUTHORITY_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_authority.v1"
FUTURE_EVALUATION_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_future_evaluation.v1"
FUTURE_AUTHORITY_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_future_authority.v1"
_AUTHORITY_NAME: Final = "_authority.json"
_MANIFEST_NAME: Final = "_manifest.json"
_EVALUATION_NAME: Final = "evaluation.json"
_MODEL_CARD_NAME: Final = "model_card.json"
_CANDIDATE_NAME: Final = "candidate.joblib"
_FUTURE_EVALUATION_NAME: Final = "future_evaluation.json"
_FUTURE_ACCESS_RESERVATION_NAME: Final = "future_access_reservation.json"
_POSITION_LEDGER_NAME: Final = "position_ledger.parquet"
_DAILY_LEDGER_NAME: Final = "daily_ledger.parquet"
_VALIDATION_PREDICTIONS_NAME: Final = "validation_predictions.parquet"


def load_complete_intraday_development_output(directory: Path) -> dict[str, Any]:
    """Strictly replay one A4.4 candidate or no-candidate authority."""

    root = directory.resolve()
    authority = _read_json(root / _AUTHORITY_NAME, "development authority")
    manifest = _read_json(root / _MANIFEST_NAME, "development manifest")
    evaluation = _read_json(root / _EVALUATION_NAME, "development evaluation")
    model_card = _read_json(root / _MODEL_CARD_NAME, "development model card")
    state = str(evaluation.get("status", ""))
    if (
        state not in {"candidate", "no_candidate"}
        or authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or manifest.get("schema_version") != MODEL_SCHEMA_VERSION
        or authority.get("state") != state
        or manifest.get("state") != state
        or authority.get("manifest_path") != _MANIFEST_NAME
        or authority.get("manifest_sha256") != file_sha256(root / _MANIFEST_NAME)
        or manifest.get("promotion_permitted") is not False
        or evaluation.get("promotion_permitted") is not False
        or model_card.get("promotion_permitted") is not False
        or evaluation.get("future_holdout_opened") is not False
        or int(evaluation.get("test_access_count", -1)) != 0
    ):
        raise DataReadinessError("A4.4 output authority identity is invalid")
    files = _object(manifest.get("files"), "development manifest files")
    expected = {
        _EVALUATION_NAME,
        _MODEL_CARD_NAME,
        _POSITION_LEDGER_NAME,
        _DAILY_LEDGER_NAME,
        _VALIDATION_PREDICTIONS_NAME,
    }
    if state == "candidate":
        expected.add(_CANDIDATE_NAME)
    if set(files) != expected:
        raise DataReadinessError("A4.4 output file inventory differs")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != expected | {_MANIFEST_NAME, _AUTHORITY_NAME}:
        raise DataReadinessError("A4.4 output immutable file set differs")
    for name, raw in files.items():
        if Path(name).name != name:
            raise DataReadinessError("A4.4 output file path is invalid")
        record = _object(raw, f"development file {name}")
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            raise DataReadinessError(f"A4.4 output file is missing: {name}")
        if int(record.get("bytes", -1)) != path.stat().st_size or record.get("sha256") != file_sha256(path):
            raise DataReadinessError(f"A4.4 output file identity failed: {name}")
    profile = _object(evaluation.get("baseline_profile"), "baseline profile")
    profile_identity = BaselineProfile(
        profile_id=str(profile.get("profile_id", "")),
        description=str(profile.get("description", "")),
        population_rule={
            str(key): _required_finite_number(value, f"population rule {key}")
            for key, value in _object(profile.get("population_rule"), "population rule").items()
        },
    )
    profile_sha256 = profile_identity.sha256()
    dataset = _object(evaluation.get("dataset"), "evaluation dataset")
    model_family = str(evaluation.get("model_family", ""))
    event_cohort = dataset.get("research_event_cohort")
    directional_event_families = {f"intraday_{subtype}_confirmed_research": subtype for subtype in DIRECTIONAL_EVENT_SUBTYPES}
    event_model = model_family == "intraday_event_confirmed_research"
    directional_subtype = directional_event_families.get(model_family)
    config_payload = _object(evaluation.get("training_config"), "training config")
    assert model_family in {
        "intraday_technical",
        "intraday_event_confirmed_research",
        *directional_event_families,
    }
    assert model_card.get("model_family") == model_family
    assert manifest.get("model_family") == model_family
    assert authority.get("model_family") == model_family
    if event_model or directional_subtype is not None:
        assert isinstance(event_cohort, dict)
        assert event_cohort.get("production_eligible") is False
        assert event_cohort.get("serving_eligible") is False
        assert event_cohort.get("future_holdout_opened") is False
        assert event_cohort.get("catalyst_role") == "confirmation_and_population_filter_not_model_feature"
        if directional_subtype is not None:
            assert event_cohort.get("event_subtype") == directional_subtype
    if model_family == "intraday_technical":
        assert event_cohort is None
    if evaluation.get("baseline_profile_sha256") != profile_sha256:
        raise RuntimeError(f"eval={evaluation.get('baseline_profile_sha256')} recreated={profile_sha256} dict={profile}")
    assert evaluation.get("baseline_profile_sha256") == profile_sha256
    assert model_card.get("baseline_profile_sha256") == profile_sha256
    assert manifest.get("baseline_profile_sha256") == profile_sha256
    assert authority.get("baseline_profile_sha256") == profile_sha256
    assert evaluation.get("training_config_sha256") == _json_sha256(config_payload)
    assert manifest.get("training_config_sha256") == evaluation.get("training_config_sha256")
    assert evaluation.get("feature_columns") == list(MODEL_FEATURE_COLUMNS)
    assert model_card.get("feature_columns") == list(MODEL_FEATURE_COLUMNS)
    assert evaluation.get("ordered_feature_sha256") == dataset.get("ordered_feature_sha256")
    assert model_card.get("ordered_feature_sha256") == dataset.get("ordered_feature_sha256")
    assert manifest.get("ordered_feature_sha256") == dataset.get("ordered_feature_sha256")
    assert authority.get("ordered_feature_sha256") == dataset.get("ordered_feature_sha256")
    assert manifest.get("dataset") == dataset
    assert authority.get("dataset_authority_sha256") == dataset.get("authority_sha256")
    config = IntradayDevelopmentConfig(**_tuple_config_values(config_payload))
    records = evaluation.get("validation_candidates")
    if not isinstance(records, list) or not records:
        raise DataReadinessError("A4.4 validation records are unavailable")
    from market_predictor.intraday.evaluation.gates import _audit_policy_choice, _evaluate_spec, _selection_key
    selected = next(
        (
            _object(record, "selected candidate")
            for record in records
            if record.get("candidate_id") == evaluation.get("selected_candidate_id")
        ),
        None,
    )
    selection_candidates = [_object(record, "selection candidate") for record in records if bool(record.get("selection_passed"))]
    selection_winner = max(selection_candidates, key=_selection_key) if selection_candidates else None
    audit_candidate, threshold, stop_threshold, passed = _audit_policy_choice(
        [_object(record, "validation candidate") for record in records],
        selected,
        preferred=selection_winner,
    )
    audit = _object(evaluation.get("auditable_policy_ledger"), "audit ledger")
    if (
        audit.get("candidate_id") != audit_candidate
        or not math.isclose(float(audit.get("threshold_bps", math.nan)), threshold)
        or not math.isclose(float(audit.get("maximum_stop_probability", math.nan)), stop_threshold)
        or audit.get("validation_passed") is not passed
    ):
        raise DataReadinessError("A4.4 selected policy replay differs")
    predictions = pd.read_parquet(root / _VALIDATION_PREDICTIONS_NAME)
    source_record = next(_object(record, "audit candidate") for record in records if record.get("candidate_id") == audit_candidate)
    hyperparameters_raw = _object(source_record.get("hyperparameters"), "hyperparameters")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in hyperparameters_raw.values()):
        raise DataReadinessError("A4.4 hyperparameters must be numeric")
    spec = _CandidateSpec(
        candidate_id=audit_candidate,
        family=str(source_record.get("family", "")),
        hyperparameters=cast(dict[str, float | int], hyperparameters_raw),
    )
    folds_raw = source_record.get("folds")
    if not isinstance(folds_raw, list):
        raise DataReadinessError("A4.4 fold evidence is unavailable")
    replayed = _evaluate_spec(
        spec,
        predictions,
        cast(list[Mapping[str, Any]], folds_raw),
        config,
        _required_finite_number(
            _object(evaluation.get("dataset"), "dataset").get("frozen_round_trip_cost_bps"),
            "dataset frozen_round_trip_cost_bps",
        ),
    )
    if _json_sha256(replayed) != _json_sha256(source_record):
        raise DataReadinessError("A4.4 validation metrics do not replay")
    if state == "candidate":
        loaded = joblib.load(root / _CANDIDATE_NAME)
        if (
            not isinstance(loaded, dict)
            or loaded.get("validation_passed") is not True
            or loaded.get("model_family") != model_family
            or loaded.get("baseline_profile_sha256") != profile_sha256
            or loaded.get("dataset") != dataset
            or loaded.get("feature_columns") != list(MODEL_FEATURE_COLUMNS)
            or loaded.get("ordered_feature_sha256") != dataset.get("ordered_feature_sha256")
            or loaded.get("training_config_sha256") != evaluation.get("training_config_sha256")
        ):
            raise DataReadinessError("A4.4 candidate payload identity differs")
    return {
        "state": state,
        "baseline_profile_sha256": profile_sha256,
        "dataset": dataset,
        "selected_candidate_id": evaluation.get("selected_candidate_id"),
        "manifest_sha256": file_sha256(root / _MANIFEST_NAME),
        "authority_sha256": file_sha256(root / _AUTHORITY_NAME),
    }


def _publish_development(
    output: Path,
    candidate: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any],
    model_card: Mapping[str, Any],
    ledger: Mapping[str, Any],
    validation_predictions: pd.DataFrame,
) -> None:
    files: dict[str, Any] = {}
    temporary = _temporary_output(output)
    try:
        if candidate is not None:
            joblib.dump(dict(candidate), temporary / _CANDIDATE_NAME, compress=3)
        _write_json(temporary / _EVALUATION_NAME, evaluation)
        _write_json(temporary / _MODEL_CARD_NAME, model_card)
        _write_ledger_files(temporary, ledger)
        validation_predictions.to_parquet(
            temporary / _VALIDATION_PREDICTIONS_NAME,
            index=False,
            compression="zstd",
        )
        for path in sorted(temporary.iterdir(), key=lambda item: item.name):
            files[path.name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        state = str(evaluation["status"])
        manifest = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "state": state,
            "model_family": evaluation["model_family"],
            "promotion_permitted": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "baseline_profile_sha256": evaluation["baseline_profile_sha256"],
            "ordered_feature_sha256": evaluation["ordered_feature_sha256"],
            "dataset": evaluation["dataset"],
            "training_config_sha256": evaluation["training_config_sha256"],
            "future_holdout_opened": False,
            "test_access_count": 0,
            "files": files,
        }
        _write_json(temporary / _MANIFEST_NAME, manifest)
        _write_json(
            temporary / _AUTHORITY_NAME,
            {
                "schema_version": AUTHORITY_SCHEMA_VERSION,
                "state": state,
                "model_family": evaluation["model_family"],
                "promotion_permitted": False,
                "manifest_path": _MANIFEST_NAME,
                "manifest_sha256": file_sha256(temporary / _MANIFEST_NAME),
                "baseline_profile_sha256": evaluation["baseline_profile_sha256"],
                "ordered_feature_sha256": evaluation["ordered_feature_sha256"],
                "dataset_authority_sha256": _object(evaluation["dataset"], "dataset")["authority_sha256"],
            },
        )
        _finish_output(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_complete_intraday_future_evaluation_output(directory: Path) -> Mapping[str, Any]:
    """Verify immutable future evidence and replay ledger-derived economics."""

    root = directory.resolve()
    expected_files = {
        _AUTHORITY_NAME,
        _MANIFEST_NAME,
        _FUTURE_ACCESS_RESERVATION_NAME,
        _FUTURE_EVALUATION_NAME,
        _POSITION_LEDGER_NAME,
        _DAILY_LEDGER_NAME,
    }
    actual_entries = {path.name for path in root.iterdir()}
    if actual_entries != expected_files:
        raise DataReadinessError("future evidence exact-file inventory differs")
    authority = _read_json(root / _AUTHORITY_NAME, "future authority")
    manifest = _read_json(root / _MANIFEST_NAME, "future manifest")
    if (
        authority.get("schema_version") != FUTURE_AUTHORITY_SCHEMA_VERSION
        or authority.get("state") != "locked_future_evaluated"
        or manifest.get("schema_version") != FUTURE_EVALUATION_SCHEMA_VERSION
        or manifest.get("state") != "locked_future_evaluated"
    ):
        raise DataReadinessError("future evidence schema or state differs")
    if authority.get("manifest_sha256") != file_sha256(root / _MANIFEST_NAME):
        raise DataReadinessError("future authority does not bind its manifest")
    files = _object(manifest.get("files"), "future manifest files")
    expected_evidence = {
        _FUTURE_ACCESS_RESERVATION_NAME,
        _FUTURE_EVALUATION_NAME,
        _POSITION_LEDGER_NAME,
        _DAILY_LEDGER_NAME,
    }
    if set(files) != expected_evidence:
        raise DataReadinessError("future manifest evidence inventory differs")
    for name, raw in files.items():
        record = _object(raw, f"future file {name}")
        path = root / name
        if record.get("sha256") != file_sha256(path) or int(record.get("bytes", -1)) != path.stat().st_size:
            raise DataReadinessError(f"future evidence identity failed: {name}")
    evaluation = _read_json(root / _FUTURE_EVALUATION_NAME, "future evaluation")
    if (
        evaluation.get("schema_version") != FUTURE_EVALUATION_SCHEMA_VERSION
        or evaluation.get("status") != "locked_future_evaluated"
        or evaluation.get("selection_changed_after_future_observation") is not False
    ):
        raise DataReadinessError("future evaluation contract differs")
    metrics = _object(evaluation.get("metrics"), "future metrics")
    positions = pd.read_parquet(root / _POSITION_LEDGER_NAME)
    daily = pd.read_parquet(root / _DAILY_LEDGER_NAME)
    if int(metrics.get("position_ledger_rows", -1)) != len(positions):
        raise DataReadinessError("future position ledger row count differs")
    if int(metrics.get("daily_ledger_rows", -1)) != len(daily):
        raise DataReadinessError("future daily ledger row count differs")
    daily_returns = daily["daily_return"].to_numpy(dtype="float64")
    replay = {
        "average_daily_net_return": float(daily_returns.mean()) if len(daily_returns) else 0.0,
        "compounded_net_return": (float(np.prod(1.0 + daily_returns) - 1.0) if len(daily_returns) else 0.0),
        "negative_session_rate": (float((daily_returns < 0.0).mean()) if len(daily_returns) else 1.0),
        "maximum_entries_per_session_observed": (int(daily["entries"].max()) if len(daily) else 0),
    }
    notionals = positions["notional"].to_numpy(dtype="float64")
    pnls = positions["pnl"].to_numpy(dtype="float64")
    replay["average_trade_net_return"] = float(pnls.sum() / notionals.sum()) if notionals.sum() > 0.0 else 0.0
    for name, expected in replay.items():
        actual = _required_finite_number(metrics.get(name), f"future metric {name}")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise DataReadinessError(f"future metric does not replay: {name}")
    for identity in ("candidate_authority_sha256", "candidate_manifest_sha256"):
        value = evaluation.get(identity)
        if not isinstance(value, str) or len(value) != 64:
            raise DataReadinessError(f"future evaluation {identity} is invalid")
    reservation = _read_json(root / _FUTURE_ACCESS_RESERVATION_NAME, "future access reservation")
    future_access = _object(evaluation.get("future_access"), "future access identity")
    candidate_authority_sha256 = str(evaluation["candidate_authority_sha256"])
    if (
        reservation.get("schema_version") != "edge_rebuild.intraday_future_access.v1"
        or reservation.get("state") != "reserved"
        or reservation.get("candidate_authority_sha256") != candidate_authority_sha256
        or future_access.get("claim_id") != candidate_authority_sha256
        or future_access.get("claim_sha256") != reservation.get("access_claim_sha256")
        or future_access.get("reservation_receipt_sha256") != file_sha256(root / _FUTURE_ACCESS_RESERVATION_NAME)
    ):
        raise DataReadinessError("future access reservation identity differs")
    for name in ("claim_id", "claim_sha256", "reservation_receipt_sha256"):
        value = future_access.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise DataReadinessError(f"future access {name} is invalid")
    _object(evaluation.get("future_dataset"), "future dataset identity")
    return evaluation


def _publish_future_evaluation(
    output: Path,
    evaluation: Mapping[str, Any],
    ledger: Mapping[str, Any],
    reservation_receipt: Path,
) -> None:
    temporary = _temporary_output(output)
    try:
        shutil.copyfile(reservation_receipt, temporary / _FUTURE_ACCESS_RESERVATION_NAME)
        _write_json(temporary / _FUTURE_EVALUATION_NAME, evaluation)
        _write_ledger_files(temporary, ledger)
        evidence_files = (
            _FUTURE_ACCESS_RESERVATION_NAME,
            _FUTURE_EVALUATION_NAME,
            _POSITION_LEDGER_NAME,
            _DAILY_LEDGER_NAME,
        )
        manifest = {
            "schema_version": FUTURE_EVALUATION_SCHEMA_VERSION,
            "state": "locked_future_evaluated",
            "promotion_permitted": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "files": {
                name: {
                    "sha256": file_sha256(temporary / name),
                    "bytes": (temporary / name).stat().st_size,
                }
                for name in evidence_files
            },
        }
        _write_json(temporary / _MANIFEST_NAME, manifest)
        _write_json(
            temporary / _AUTHORITY_NAME,
            {
                "schema_version": FUTURE_AUTHORITY_SCHEMA_VERSION,
                "state": "locked_future_evaluated",
                "manifest_path": _MANIFEST_NAME,
                "manifest_sha256": file_sha256(temporary / _MANIFEST_NAME),
            },
        )
        load_complete_intraday_future_evaluation_output(temporary)
        _finish_output(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_validation_passed_candidate(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    replay = load_complete_intraday_development_output(directory)
    if replay["state"] != "candidate":
        raise DataReadinessError("future holdout is locked until validation publishes a candidate")
    authority = _read_json(directory / _AUTHORITY_NAME, "candidate authority")
    manifest = _read_json(directory / _MANIFEST_NAME, "candidate manifest")
    if authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION:
        raise DataReadinessError("future evaluation accepts only A4.4 bar-baseline authorities")
    if authority.get("state") != "candidate" or manifest.get("state") != "candidate":
        raise DataReadinessError("future holdout is locked until validation publishes a candidate")
    if authority.get("manifest_sha256") != file_sha256(directory / _MANIFEST_NAME):
        raise DataReadinessError("candidate authority does not bind its manifest")
    files = _object(manifest.get("files"), "candidate manifest files")
    for name, raw in files.items():
        record = _object(raw, f"candidate file {name}")
        path = directory / str(name)
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            raise DataReadinessError(f"candidate file identity failed: {name}")
    if _CANDIDATE_NAME not in files:
        raise DataReadinessError("validation-passed candidate model is absent")
    loaded = joblib.load(directory / _CANDIDATE_NAME)
    if not isinstance(loaded, dict) or loaded.get("validation_passed") is not True:
        raise DataReadinessError("future holdout is locked until validation passes")
    return loaded, manifest


def _temporary_output(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))


def _require_output_isolated(output: Path, *inputs: Path) -> None:
    target = output.resolve()
    for immutable_input in inputs:
        source = immutable_input.resolve()
        if target == source or target in source.parents or source in target.parents:
            raise DataReadinessError("output overlaps an immutable input authority")
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")


def _require_registry_isolated(registry_directory: Path, *protected_paths: Path) -> None:
    registry = registry_directory.expanduser().resolve()
    for protected_path in protected_paths:
        protected = protected_path.resolve()
        if registry == protected or registry in protected.parents or protected in registry.parents:
            raise DataReadinessError("future access registry overlaps an immutable input or output")


def _reserve_future_access(
    candidate: Path,
    future_dataset: Path,
    registry_directory: Path,
) -> Path:
    """Atomically reserve one candidate's future holdout before future data is read."""

    candidate_authority = candidate / _AUTHORITY_NAME
    candidate_authority_sha256 = file_sha256(candidate_authority)
    registry = registry_directory.expanduser().resolve()
    registry.mkdir(parents=True, exist_ok=True)
    claim = registry / f"{candidate_authority_sha256}.claim"
    reservation_receipt = registry / f"{candidate_authority_sha256}.reservation.json"
    try:
        claim.touch(exist_ok=False)
    except FileExistsError:
        raise DataReadinessError("future holdout access was already consumed") from None
    try:
        payload = {
            "schema_version": "edge_rebuild.intraday_future_access.v1",
            "state": "reserved",
            "access_claim_sha256": file_sha256(claim),
            "candidate_authority_sha256": candidate_authority_sha256,
            "future_dataset_directory": str(future_dataset.resolve()),
            "reserved_at_utc": datetime.now(UTC).isoformat(),
        }
        _write_exclusive_json(reservation_receipt, payload)
    except BaseException as exc:
        _record_future_access_failure(claim, exc)
        raise
    return reservation_receipt


def _record_future_access_failure(access_lock: Path, error: BaseException) -> Path:
    """Publish immutable failure evidence without releasing the one-time reservation."""

    failure_receipt = access_lock.with_name(f"{access_lock.stem}.failure.json")
    payload = {
        "schema_version": "edge_rebuild.intraday_future_access_failure.v1",
        "state": "failed_after_reservation",
        "access_lock_sha256": file_sha256(access_lock),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "failed_at_utc": datetime.now(UTC).isoformat(),
    }
    try:
        with failure_receipt.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError:
        raise DataReadinessError("future holdout failure receipt already exists") from error
    return failure_receipt


def _future_access_identity(reservation_receipt: Path) -> dict[str, str]:
    reservation = _read_json(reservation_receipt, "future access reservation")
    candidate_authority_sha256 = reservation.get("candidate_authority_sha256")
    claim_sha256 = reservation.get("access_claim_sha256")
    if (
        reservation.get("schema_version") != "edge_rebuild.intraday_future_access.v1"
        or reservation.get("state") != "reserved"
        or not isinstance(candidate_authority_sha256, str)
        or len(candidate_authority_sha256) != 64
        or not isinstance(claim_sha256, str)
        or len(claim_sha256) != 64
    ):
        raise DataReadinessError("future access reservation is invalid")
    claim = reservation_receipt.with_name(f"{candidate_authority_sha256}.claim")
    if not claim.is_file() or file_sha256(claim) != claim_sha256:
        raise DataReadinessError("future access claim identity differs")
    return {
        "claim_id": candidate_authority_sha256,
        "claim_sha256": claim_sha256,
        "reservation_receipt_sha256": file_sha256(reservation_receipt),
    }


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def _write_ledger_files(directory: Path, ledger: Mapping[str, Any]) -> None:
    positions = ledger.get("position_records")
    daily = ledger.get("daily_records")
    if not isinstance(positions, list) or not isinstance(daily, list):
        raise DataReadinessError("portfolio ledger records are unavailable")
    pd.DataFrame(positions).to_parquet(directory / _POSITION_LEDGER_NAME, index=False)
    pd.DataFrame(daily).to_parquet(directory / _DAILY_LEDGER_NAME, index=False)


def _finish_output(temporary: Path, output: Path) -> None:
    try:
        temporary.rename(output)
    except FileExistsError:
        raise FileExistsError(f"immutable output already exists: {output}") from None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataReadinessError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"{label} is invalid") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"{label} must be an object")
    return value


def _dataset_identity(published: PublishedIntradayDataset) -> dict[str, Any]:
    return {
        "dataset_sha256": published.dataset_sha256,
        "manifest_sha256": published.manifest_sha256,
        "authority_sha256": published.authority_sha256,
        "request_sha256": published.request_sha256,
        "transformation_sha256": published.transformation_sha256,
        "session_unit_inventory_sha256": published.session_unit_inventory_sha256,
        "ordered_feature_sha256": published.ordered_feature_sha256,
        "strategy_contract_sha256": published.strategy_contract_sha256,
        "frozen_round_trip_cost_bps": published.frozen_round_trip_cost_bps,
    }


def _gate_contract(config: IntradayDevelopmentConfig) -> dict[str, Any]:
    return {
        "minimum_scope_rows": config.minimum_scope_rows,
        "minimum_scope_securities": config.minimum_scope_securities,
        "minimum_positive_net_return_roc_auc": config.minimum_positive_net_return_roc_auc,
        "minimum_seen_positive_net_lift": config.minimum_seen_positive_net_lift,
        "minimum_unseen_positive_net_lift": config.minimum_unseen_positive_net_lift,
        "minimum_seen_stop_hit_roc_auc": config.minimum_seen_stop_hit_roc_auc,
        "minimum_unseen_stop_hit_roc_auc": config.minimum_unseen_stop_hit_roc_auc,
        "maximum_stop_hit_brier": config.maximum_stop_hit_brier,
        "maximum_stop_hit_ece": config.maximum_stop_hit_ece,
        "minimum_stop_hit_brier_skill": 0.0,
        "minimum_validation_trades": config.minimum_validation_trades,
        "minimum_validation_sessions_with_trades": config.minimum_validation_sessions_with_trades,
        "minimum_average_trade_net_return_bps": config.minimum_average_trade_net_return_bps,
        "minimum_average_daily_net_return_bps": config.minimum_average_daily_net_return_bps,
        "minimum_daily_return_ci_low_bps": config.minimum_daily_return_ci_low_bps,
        "minimum_profit_factor": config.minimum_profit_factor,
        "minimum_economic_rank_gain_bps": config.minimum_economic_rank_gain_bps,
        "minimum_average_spy_excess_bps": config.minimum_average_spy_excess_bps,
        "minimum_average_qqq_excess_bps": config.minimum_average_qqq_excess_bps,
        "minimum_average_sector_excess_bps": config.minimum_average_sector_excess_bps,
        "maximum_drawdown": config.maximum_drawdown,
        "maximum_round_trip_turnover": config.maximum_round_trip_turnover,
        "minimum_profitable_fold_fraction": config.minimum_profitable_fold_fraction,
        "maximum_negative_session_rate": config.maximum_negative_session_rate,
        "minimum_return_to_drawdown": config.minimum_return_to_drawdown,
        "maximum_entries_per_decision": config.maximum_candidates_per_decision,
        "maximum_concurrent_positions": config.maximum_concurrent_positions,
        "stress_cost_bps": config.stress_cost_bps,
        "minimum_stress_average_daily_return_bps": config.minimum_stress_average_daily_return_bps,
    }


def _future_data_contract(config: IntradayDevelopmentConfig) -> dict[str, Any]:
    return {
        "development_end_date": config.development_end_date,
        "minimum_session_date": config.future_holdout_start_date,
        "minimum_sessions": config.minimum_validation_sessions,
        "minimum_rows": config.minimum_rows,
        "minimum_securities": config.minimum_securities,
        "required_timeframe": "1Min",
        "required_price_feed": "sip",
        "required_adjustment": "all",
        "future_access_registry_directory": str(_resolved_future_access_registry(config)),
        "selection_must_remain_frozen": True,
    }


def _resolved_future_access_registry(config: IntradayDevelopmentConfig) -> Path:
    configured = Path(config.future_access_registry_directory).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    return (Path.home() / ".market-predictor" / configured).resolve()


def _tuple_config_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(payload)
    for name in (
        "expected_net_return_thresholds_bps",
        "maximum_stop_probability_thresholds",
        "ridge_alphas",
        "logistic_c_values",
        "hgb_learning_rates",
        "hgb_max_leaf_nodes",
        "cost_curve_bps",
    ):
        values[name] = tuple(values[name])
    return values


def _guard_memory(config: IntradayDevelopmentConfig, stage: str, *, peak: bool) -> None:
    assert_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )
    if peak:
        assert_peak_memory_budget(
            hard_budget_gib=config.maximum_process_memory_gib,
            headroom_gib=config.memory_guard_headroom_gib,
            stage=stage,
        )


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO date") from exc


def _required_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataReadinessError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DataReadinessError(f"{label} must be finite")
    return result


def _strict_bool(value: Any) -> bool:
    return value is True or isinstance(value, np.bool_) and bool(value)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DataReadinessError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}
