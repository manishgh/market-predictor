from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild import intraday_training
from market_predictor.edge_rebuild.intraday_dataset import INTRADAY_DATASET_AUTHORITY_SCHEMA
from market_predictor.edge_rebuild.intraday_features import FEATURE_SCHEMA_VERSION
from market_predictor.edge_rebuild.intraday_history import json_sha256
from market_predictor.edge_rebuild.intraday_labels import LABEL_SCHEMA_VERSION
from market_predictor.edge_rebuild.intraday_training import (
    DATASET_SCHEMA_VERSION,
    MODEL_FEATURE_COLUMNS,
    IntradayTrainingConfig,
    load_intraday_training_config,
    load_published_intraday_dataset,
    train_intraday_edge_candidate,
)
from market_predictor.v3.errors import DataReadinessError


def test_trains_sequential_candidates_and_publishes_immutable_evidence(tmp_path: Path) -> None:
    authority = _publish_dataset(tmp_path / "dataset", _training_frame())
    output = tmp_path / "candidate"

    result = train_intraday_edge_candidate(authority, output, config=_config())

    assert result.output_directory == output
    assert set(path.name for path in output.iterdir()) == {
        "candidate.joblib",
        "evaluation.json",
        "model_card.json",
        "_manifest.json",
        "_authority.json",
    }
    evaluation = _read_json(output / "evaluation.json")
    assert evaluation["selection_basis"] == "validation_only"
    assert evaluation["selection_policy"]["name"] == "ER5_CONSERVATIVE_ECONOMICS_V1"
    assert evaluation["selection_policy"]["auc_used_for_selection"] is False
    assert evaluation["test_access_count"] == 1
    assert evaluation["training_config_sha256"]
    assert evaluation["training_config"]["top_k"] == 3
    assert evaluation["split"]["random_or_row_split_used"] is False
    assert evaluation["split"]["split_unit"] == "session_date_et"
    assert evaluation["split"]["overnight_embargo_sessions"] == 1
    assert len(evaluation["validation_candidates"]) == 4
    assert {candidate["family"] for candidate in evaluation["validation_candidates"]} == {
        "deterministic",
        "logistic",
        "hist_gradient_boosting",
        "xgb_ranker",
    }
    metrics = evaluation["final_test"]
    assert {
        "roc_auc",
        "pr_auc",
        "brier_score",
        "calibration_bias",
        "expected_calibration_error",
        "top_k_average_gross_return",
        "top_k_average_net_return",
        "top_k_average_spy_excess_return",
        "top_k_average_sector_excess_return",
        "top_k_win_rate_after_costs",
        "turnover",
        "trade_count",
        "max_drawdown_after_costs",
        "profit_factor_after_costs",
        "daily_session_block_rank_ic_mean",
        "ndcg_at_k",
        "top_minus_bottom_net_return_spread",
        "session_block_bootstrap_95_ci",
        "by_year",
        "by_session_segment",
    }.issubset(metrics)
    assert metrics["top_k_average_net_return"] == pytest.approx(
        metrics["top_k_average_gross_return"] - 0.001,
        abs=1e-12,
    )
    assert all(
        pd.Timestamp(fold["max_train_label_available_at_utc"]) < pd.Timestamp(fold["min_validation_decision_time_utc"])
        for candidate in evaluation["validation_candidates"]
        for fold in candidate["folds"]
    )
    ranker = next(candidate for candidate in evaluation["validation_candidates"] if candidate["family"] == "xgb_ranker")
    assert ranker["hyperparameters"]["objective"] == "rank:ndcg"
    assert ranker["hyperparameters"]["n_jobs"] == 1
    assert ranker["hyperparameters"]["library_version"] == "3.3.0"
    for candidate in evaluation["validation_candidates"]:
        assert "temporal_validation" in candidate
        assert "unseen_security_validation" in candidate
        assert candidate["selection_key"] == list(intraday_training._validation_selection_key(candidate))

    model_card = _read_json(output / "model_card.json")
    assert model_card["status"] == "candidate"
    assert model_card["promotion_permitted"] is False
    payload = joblib.load(output / "candidate.joblib")
    assert payload["status"] == "candidate"
    assert payload["promotion_permitted"] is False

    manifest = _read_json(output / "_manifest.json")
    authority_record = _read_json(output / "_authority.json")
    assert authority_record["manifest_sha256"] == file_sha256(output / "_manifest.json")
    for name, record in manifest["files"].items():
        assert record["sha256"] == file_sha256(output / name)

    with pytest.raises(FileExistsError, match="immutable output"):
        train_intraday_edge_candidate(authority, output, config=_config())


def test_repository_training_policy_is_complete_and_frozen() -> None:
    config = load_intraday_training_config(Path("configs/edge_rebuild_intraday_training.toml"))

    assert config.validation_folds == 3
    assert config.maximum_label_horizon_minutes == 30
    assert config.top_k == 10
    assert config.maximum_process_memory_gib == 4.0
    learned = sum(spec.family != "deterministic" for spec in intraday_training._candidate_specs(config))
    assert learned == 5
    assert learned <= config.maximum_learned_candidates


def test_trainer_uses_only_normalized_causal_price_features() -> None:
    expected_new_features = {
        "atr_fraction_of_close",
        "normalized_volume_overshoot",
        "volume_bar_duration_minutes",
        "relative_volume_at_activation",
        "minutes_since_causal_activation",
        "regular_session_progress",
    }
    raw_price_features = {
        "open",
        "high",
        "low",
        "close",
        "atr_14",
        "stock_clock_context_close",
        "stock_clock_session_vwap",
    }

    assert expected_new_features.issubset(MODEL_FEATURE_COLUMNS)
    assert raw_price_features.isdisjoint(MODEL_FEATURE_COLUMNS)
    assert not any(
        token in column
        for column in MODEL_FEATURE_COLUMNS
        for token in ("news", "catalyst", "sentiment", "sec_filing", "source_count_sec")
    )
    assert len(intraday_training._candidate_specs(_config())) - 1 <= 6


def test_loaded_model_features_are_finite_float32(tmp_path: Path) -> None:
    published = load_published_intraday_dataset(_publish_dataset(tmp_path / "dataset", _training_frame()))

    assert all(published.frame[column].dtype == np.dtype("float32") for column in MODEL_FEATURE_COLUMNS)
    assert np.isfinite(published.frame.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(dtype="float32")).all()


def test_temporal_test_poison_cannot_change_validation_selection(tmp_path: Path) -> None:
    baseline = _training_frame()
    poisoned = baseline.copy()
    final_sessions = sorted(poisoned["session_date_et"].astype(str).unique())[-16:]
    future = poisoned["session_date_et"].astype(str).isin(final_sessions)
    poisoned.loc[future, "target"] = 1 - poisoned.loc[future, "target"]
    poisoned.loc[future, "target_hit"] = poisoned.loc[future, "target"].astype(bool)
    poisoned.loc[future, "gross_return"] *= -1.0
    poisoned.loc[future, "net_return"] = poisoned.loc[future, "gross_return"] - 0.001
    poisoned.loc[future, "spy_excess_return"] = poisoned.loc[future, "net_return"] - 0.0002
    poisoned.loc[future, "sector_excess_return"] = poisoned.loc[future, "net_return"] - 0.0001

    first = train_intraday_edge_candidate(
        _publish_dataset(tmp_path / "baseline", baseline),
        tmp_path / "baseline_candidate",
        config=_config(),
    )
    second = train_intraday_edge_candidate(
        _publish_dataset(tmp_path / "poisoned", poisoned),
        tmp_path / "poisoned_candidate",
        config=_config(),
    )

    assert first.selected_candidate_id == second.selected_candidate_id
    assert first.evaluation["validation_candidates"] == second.evaluation["validation_candidates"]
    assert first.evaluation["final_test"] != second.evaluation["final_test"]
    assert first.evaluation["test_access_count"] == second.evaluation["test_access_count"] == 1


def test_selection_uses_conservative_economics_not_auc() -> None:
    high_auc_bad_economics = _selection_record(
        "high_auc",
        roc_auc=0.95,
        benchmark_ci_low=-0.002,
        net_ci_low=-0.001,
    )
    lower_auc_good_economics = _selection_record(
        "good_economics",
        roc_auc=0.55,
        benchmark_ci_low=0.001,
        net_ci_low=0.0005,
    )

    selected = intraday_training._select_from_validation([high_auc_bad_economics, lower_auc_good_economics])

    assert selected["candidate_id"] == "good_economics"


def test_reports_security_overlap_without_row_or_session_overlap(tmp_path: Path) -> None:
    authority = _publish_dataset(tmp_path / "dataset", _training_frame())
    result = train_intraday_edge_candidate(authority, tmp_path / "candidate", config=_config())

    audit = result.evaluation["overlap_audit"]
    assert audit["row_identity_overlap_total"] == 0
    assert audit["session_date_overlap_total"] == 0
    assert audit["security_overlap_is_reported_not_forbidden"] is True
    assert audit["final"]["security_overlap"] == 8
    assert audit["final"]["security_overlap_fraction_of_right"] == 1.0
    assert audit["all_rows_from_one_session_stay_in_one_temporal_split"] is True
    assert audit["development_security_holdout"]["security_count"] == 2
    assert audit["development_security_holdout"]["fit_holdout_security_overlap"] == 0
    assert audit["development_security_holdout"]["separate_from_locked_test"] is True
    for fold in audit["folds"]:
        assert fold["embargo_session_overlap_with_train"] == 0
        assert fold["embargo_session_overlap_with_validation"] == 0


def test_security_holdout_is_deterministic_and_order_independent() -> None:
    frame = _training_frame()
    frame["session_date_et"] = frame["session_date_et"].astype(str)
    development_sessions = tuple(sorted(frame["session_date_et"].unique())[:63])

    first = intraday_training._deterministic_security_holdout(frame, development_sessions)
    second = intraday_training._deterministic_security_holdout(
        frame.sample(frac=1.0, random_state=91),
        development_sessions,
    )

    assert first == second
    assert len(first) == 2


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        (
            "feature_available_at_utc",
            lambda row: row["decision_time_utc"] + pd.Timedelta(seconds=1),
            "feature availability",
        ),
        (
            "label_available_at_utc",
            lambda row: row["decision_time_utc"],
            "label availability",
        ),
    ],
)
def test_rejects_feature_or_label_availability_leakage(
    tmp_path: Path,
    column: str,
    replacement: Any,
    message: str,
) -> None:
    frame = _training_frame()
    frame.loc[0, column] = replacement(frame.loc[0])
    authority = _publish_dataset(tmp_path / "dataset", frame)

    with pytest.raises(DataReadinessError, match=message):
        train_intraday_edge_candidate(authority, tmp_path / "candidate", config=_config())


def test_purges_labels_that_are_not_available_before_validation(tmp_path: Path) -> None:
    frame = _training_frame()
    delayed_group = frame["decision_group_id"].drop_duplicates().iloc[10]
    delayed = frame["decision_group_id"].eq(delayed_group)
    frame.loc[delayed, "label_available_at_utc"] = pd.Timestamp("2099-01-01", tz="UTC")
    authority = _publish_dataset(tmp_path / "dataset", frame)

    result = train_intraday_edge_candidate(authority, tmp_path / "candidate", config=_config())

    for candidate in result.evaluation["validation_candidates"]:
        assert [fold["train_sessions"] for fold in candidate["folds"]] == [10, 35]


def test_each_complete_session_has_one_fold_and_one_overnight_embargo(tmp_path: Path) -> None:
    result = train_intraday_edge_candidate(
        _publish_dataset(tmp_path / "dataset", _training_frame()),
        tmp_path / "candidate",
        config=_config(),
    )

    for candidate in result.evaluation["validation_candidates"]:
        for fold in candidate["folds"]:
            train = set(fold["train_session_dates"])
            validation = set(fold["validation_session_dates"])
            embargo = set(fold["embargo_session_dates"])
            assert len(embargo) == 1
            assert train.isdisjoint(validation)
            assert train.isdisjoint(embargo)
            assert validation.isdisjoint(embargo)


def test_rejects_cost_omission_or_mismatch(tmp_path: Path) -> None:
    frame = _training_frame()
    frame["net_return"] = frame["gross_return"]
    authority = _publish_dataset(tmp_path / "dataset", frame)

    with pytest.raises(DataReadinessError, match="frozen round-trip cost"):
        train_intraday_edge_candidate(authority, tmp_path / "candidate", config=_config())


def test_rejects_label_path_beyond_frozen_thirty_minutes(tmp_path: Path) -> None:
    frame = _training_frame()
    frame.loc[0, "exit_bar_end_utc"] = frame.loc[0, "entry_time_utc"] + pd.Timedelta(minutes=31)
    frame.loc[0, "label_available_at_utc"] = frame.loc[0, "exit_bar_end_utc"]

    with pytest.raises(DataReadinessError, match="30-minute horizon"):
        train_intraday_edge_candidate(
            _publish_dataset(tmp_path / "dataset", frame),
            tmp_path / "candidate",
            config=_config(),
        )


@pytest.mark.parametrize("corruption", ["legacy_schema", "manifest_hash", "dataset_hash", "unpublished"])
def test_rejects_invalid_or_untrusted_authority(tmp_path: Path, corruption: str) -> None:
    authority = _publish_dataset(tmp_path / "dataset", _training_frame())
    authority_path = authority / "_authority.json"
    manifest_path = authority / "_manifest.json"
    authority_record = _read_json(authority_path)
    manifest = _read_json(manifest_path)
    if corruption == "legacy_schema":
        authority_record["schema"] = "intraday.dataset.legacy"
        _write_json(authority_path, authority_record)
    elif corruption == "manifest_hash":
        authority_record["artifact_sha256"] = "0" * 64
        _write_json(authority_path, authority_record)
    elif corruption == "dataset_hash":
        manifest["partitions"][0]["sha256"] = "f" * 64
        for record in manifest["files"]:
            if record["path"].endswith(".parquet"):
                record["sha256"] = "f" * 64
        _write_json(manifest_path, manifest)
        authority_record["artifact_sha256"] = file_sha256(manifest_path)
        _write_json(authority_path, authority_record)
    else:
        authority_record["state"] = "draft"
        _write_json(authority_path, authority_record)

    with pytest.raises(DataReadinessError):
        load_published_intraday_dataset(authority)


def test_memory_guard_stops_before_any_artifact_is_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _publish_dataset(tmp_path / "dataset", _training_frame())
    output = tmp_path / "candidate"
    calls: list[tuple[float, float, str]] = []

    def reject_memory(*, hard_budget_gib: float, headroom_gib: float, stage: str) -> None:
        calls.append((hard_budget_gib, headroom_gib, stage))
        raise DataReadinessError("memory pressure")

    monkeypatch.setattr(intraday_training, "assert_memory_budget", reject_memory)
    with pytest.raises(DataReadinessError, match="memory pressure"):
        train_intraday_edge_candidate(authority, output, config=_config())
    assert calls == [(4.0, 0.75, "intraday training start")]
    assert not output.exists()

    with pytest.raises(ValueError, match=r"\(0, 4\] GiB"):
        IntradayTrainingConfig(maximum_process_memory_gib=4.01)
    with pytest.raises(ValueError, match="candidate budget"):
        IntradayTrainingConfig(maximum_learned_candidates=7)


def test_negative_deterministic_economics_do_not_veto_model_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _training_frame()
    frame["gross_return"] = -0.002
    frame["net_return"] = -0.003
    frame["spy_excess_return"] = -0.0032
    frame["sector_excess_return"] = -0.0031
    authority = _publish_dataset(tmp_path / "dataset", frame)
    fitted_families: list[str] = []
    original = intraday_training._fit_candidate

    def recording_fit(*args: Any, **kwargs: Any) -> Any:
        fitted_families.append(args[0].family)
        return original(*args, **kwargs)

    monkeypatch.setattr(intraday_training, "_fit_candidate", recording_fit)
    train_intraday_edge_candidate(authority, tmp_path / "candidate", config=_config())
    assert "logistic" in fitted_families
    assert "hist_gradient_boosting" in fitted_families
    assert "xgb_ranker" in fitted_families


def _config() -> IntradayTrainingConfig:
    return IntradayTrainingConfig(
        validation_folds=2,
        final_test_fraction=0.20,
        minimum_train_sessions=12,
        minimum_validation_sessions=4,
        embargo_sessions=1,
        maximum_label_horizon_minutes=30,
        calibration_fraction=0.20,
        minimum_calibration_sessions=2,
        minimum_rows=100,
        minimum_securities=4,
        top_k=3,
        logistic_c_values=(1.0,),
        hgb_learning_rates=(0.05,),
        hgb_max_leaf_nodes=(7,),
        hgb_max_iter=20,
        hgb_max_bins=31,
        ranker_learning_rates=(0.1,),
        ranker_max_depths=(2,),
        ranker_n_estimators=10,
        ranker_max_bin=31,
        bootstrap_samples=100,
        maximum_process_memory_gib=4.0,
        memory_guard_headroom_gib=0.75,
    )


def _training_frame() -> pd.DataFrame:
    rng = np.random.default_rng(19)
    sessions = pd.bdate_range("2024-01-02", periods=80)
    rows: list[dict[str, Any]] = []
    for session_index, session in enumerate(sessions):
        market = 0.4 * math.sin(session_index / 6.0)
        for decision_index, offset_minutes in enumerate((0, 120)):
            decision = pd.Timestamp(session, tz="America/New_York") + pd.Timedelta(hours=11, minutes=offset_minutes)
            decision = decision.tz_convert("UTC")
            group_id = decision.isoformat()
            for security_index in range(8):
                feature_1 = float(rng.normal() + 0.3 * market + decision_index * 0.05)
                feature_2 = float(rng.normal() - 0.2 * market)
                raw = 1.1 * feature_1 - 0.65 * feature_2 + 0.15 * ((security_index + session_index) % 3 - 1)
                probability = 1.0 / (1.0 + math.exp(-raw))
                target = int((security_index + session_index + decision_index) % 8 < round(8 * probability))
                gross = (0.003 if target else -0.0015) + float(rng.normal(0.0, 0.0002))
                net = gross - 0.001
                model_features = {
                    column: feature_1 + (index % 3 - 1) * 0.05 if index % 2 == 0 else feature_2 + (index % 5 - 2) * 0.03
                    for index, column in enumerate(MODEL_FEATURE_COLUMNS)
                }
                model_features["return_1_bar"] = feature_1
                rows.append(
                    {
                        "dataset_row_id": f"{group_id}|SEC{security_index:02d}",
                        "ticker": f"T{security_index:02d}",
                        "security_id": f"SEC{security_index:02d}",
                        "session_date_et": session.date(),
                        "decision_group_id": group_id,
                        "decision_time_utc": decision,
                        "feature_available_at_utc": decision,
                        "entry_time_utc": decision + pd.Timedelta(minutes=1),
                        "exit_bar_end_utc": decision + pd.Timedelta(minutes=31),
                        "label_available_at_utc": decision + pd.Timedelta(minutes=31),
                        "feature_eligible": True,
                        "label_eligible": True,
                        "dataset_eligible": True,
                        "session_segment": "midday" if decision_index == 0 else "late",
                        "sector": "Information Technology" if security_index % 2 == 0 else "Health Care",
                        "market_cap_bucket": "large" if security_index < 4 else "mid",
                        "feature_1": feature_1,
                        "feature_2": feature_2,
                        "deterministic_score": probability,
                        "target": target,
                        "target_hit": bool(target),
                        "gross_return": gross,
                        "cost": 0.001,
                        "net_return": net,
                        "spy_excess_return": net - 0.0002,
                        "sector_excess_return": net - 0.0001,
                        **model_features,
                    }
                )
    return pd.DataFrame(rows)


def _publish_dataset(directory: Path, frame: pd.DataFrame) -> Path:
    directory.mkdir(parents=True)
    parent_lineage = {"selection_manifest_sha256": "a" * 64, "strategy_contract_sha256": "b" * 64}
    parent_lineage_sha256 = json_sha256(parent_lineage)
    prepared = frame.copy()
    if "volume_bar_number" not in prepared.columns:
        prepared["volume_bar_number"] = prepared.groupby(["session_date_et", "ticker"], sort=False, observed=True).cumcount() + 1
    prepared["session_month_et"] = pd.to_datetime(prepared["session_date_et"], errors="raise").dt.strftime("%Y-%m")
    expected_by_month = {
        str(month): int(len(rows.loc[:, ["session_date_et", "ticker"]].drop_duplicates()))
        for month, rows in prepared.groupby("session_month_et", sort=True, observed=True)
    }
    request_payload = {
        "schema": DATASET_SCHEMA_VERSION,
        "parent_lineage": parent_lineage,
        "parent_lineage_sha256": parent_lineage_sha256,
        "strategy_contract_sha256": "b" * 64,
        "partitioning": ["session_month_et"],
        "partition_layout": "one_parquet_file_per_calendar_month",
        "partition_row_group": "one_completed_exchange_session",
        "expected_selected_stock_sessions_by_month": expected_by_month,
        "expected_usable_stock_sessions_by_month": expected_by_month,
    }
    request_sha256 = json_sha256(request_payload)
    request_path = directory / "_request.json"
    _write_json(request_path, {**request_payload, "request_sha256": request_sha256})
    partition_records: list[dict[str, Any]] = []
    canonical_schema: pa.Schema | None = None
    for month, month_rows in prepared.groupby("session_month_et", sort=True, observed=True):
        month_rows = month_rows.drop(columns="session_month_et")
        partition_path = directory / "partitions" / f"session_month_et={month}" / "part-00000.parquet"
        partition_path.parent.mkdir(parents=True)
        writer: pq.ParquetWriter | None = None
        try:
            for _, session_rows in month_rows.groupby("session_date_et", sort=True, observed=True):
                ordered = session_rows.sort_values(
                    ["session_date_et", "ticker", "volume_bar_number"],
                    kind="stable",
                ).reset_index(drop=True)
                table = pa.Table.from_pandas(ordered, preserve_index=False).replace_schema_metadata(None)
                if canonical_schema is None:
                    canonical_schema = table.schema
                else:
                    assert table.schema.equals(canonical_schema)
                if writer is None:
                    writer = pq.ParquetWriter(
                        partition_path,
                        canonical_schema,
                        compression="zstd",
                        use_dictionary=True,
                        write_statistics=True,
                    )
                writer.write_table(table, row_group_size=len(table))
        finally:
            if writer is not None:
                writer.close()
        sessions = pd.to_datetime(month_rows["session_date_et"], errors="raise").dt.date
        partition_records.append(
            {
                "path": partition_path.relative_to(directory).as_posix(),
                "sha256": file_sha256(partition_path),
                "bytes": partition_path.stat().st_size,
                "rows": len(month_rows),
                "eligible_rows": int(month_rows["dataset_eligible"].sum()),
                "session_month_et": str(month),
                "first_session_date_et": sessions.min().isoformat(),
                "last_session_date_et": sessions.max().isoformat(),
                "stock_sessions": int(len(month_rows.loc[:, ["session_date_et", "ticker"]].drop_duplicates())),
                "ticker_count": int(month_rows["ticker"].nunique()),
            }
        )
    request_record = {
        "path": request_path.name,
        "sha256": file_sha256(request_path),
        "bytes": request_path.stat().st_size,
        "rows": 1,
    }
    audit_directory = directory / "audit"
    audit_directory.mkdir()
    pair_audit = prepared.loc[:, ["ticker", "session_date_et"]].drop_duplicates()
    pair_audit = pair_audit.assign(
        status="published",
        reason=pd.NA,
        source_rows=1,
        completed_volume_bars=1,
        feature_rows=1,
        feature_eligible_rows=1,
        label_eligible_rows=1,
        dataset_eligible_rows=1,
        abstention_rows=0,
    )
    pair_path = audit_directory / "stock_session_audit.parquet"
    pair_audit.to_parquet(pair_path, index=False)
    abstention_path = audit_directory / "abstentions.parquet"
    pd.DataFrame(
        columns=[
            "dataset_row_id",
            "ticker",
            "session_date_et",
            "volume_bar_number",
            "feature_available_at_utc",
            "stage",
            "reason",
        ]
    ).to_parquet(abstention_path, index=False)
    audit_records = [
        {
            "path": pair_path.relative_to(directory).as_posix(),
            "sha256": file_sha256(pair_path),
            "bytes": pair_path.stat().st_size,
            "rows": len(pair_audit),
        },
        {
            "path": abstention_path.relative_to(directory).as_posix(),
            "sha256": file_sha256(abstention_path),
            "bytes": abstention_path.stat().st_size,
            "rows": 0,
        },
    ]
    manifest = {
        "schema": DATASET_SCHEMA_VERSION,
        "status": "complete",
        "request_sha256": request_sha256,
        "parent_lineage": parent_lineage,
        "parent_lineage_sha256": parent_lineage_sha256,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "partitioning": request_payload["partitioning"],
        "partition_layout": request_payload["partition_layout"],
        "partition_row_group": request_payload["partition_row_group"],
        "partitions": partition_records,
        "files": [*partition_records, *audit_records, request_record],
        "summary": {
            "rows": len(frame),
            "dataset_eligible_rows": int(frame["dataset_eligible"].sum()),
            "selected_stock_sessions": sum(expected_by_month.values()),
            "excluded_stock_sessions": 0,
            "published_stock_sessions": sum(expected_by_month.values()),
            "abstention_rows": 0,
        },
        "training_contract": {
            "eligibility_column": "dataset_eligible",
            "feature_columns_exclude": [
                "target_hit",
                "gross_return",
                "net_return",
                "spy_excess_return",
                "sector_excess_return",
                "rank_label",
            ],
        },
    }
    manifest_path = directory / "_manifest.json"
    _write_json(manifest_path, manifest)
    authority = {
        "schema": INTRADAY_DATASET_AUTHORITY_SCHEMA,
        "state": "complete",
        "artifact": "_manifest.json",
        "artifact_sha256": file_sha256(manifest_path),
        "request_sha256": request_sha256,
        "parent_lineage_sha256": parent_lineage_sha256,
        "partitions": len(partition_records),
        "rows": len(frame),
    }
    _write_json(directory / "_authority.json", authority)
    for record in partition_records:
        parquet = pq.ParquetFile(directory / str(record["path"]))
        month_rows = prepared.loc[prepared["session_month_et"].eq(record["session_month_et"])]
        assert parquet.metadata.num_row_groups == month_rows["session_date_et"].nunique()
    return directory


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _selection_record(
    candidate_id: str,
    *,
    roc_auc: float,
    benchmark_ci_low: float,
    net_ci_low: float,
) -> dict[str, Any]:
    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": roc_auc,
        "top_k_average_net_return": net_ci_low + 0.001,
        "top_k_average_spy_excess_return": benchmark_ci_low + 0.001,
        "top_k_average_sector_excess_return": benchmark_ci_low + 0.001,
        "max_drawdown_after_costs": 0.02,
        "expected_calibration_error": 0.05,
        "brier_score": 0.20,
        "decision_group_rank_ic_mean": 0.1,
        "session_block_bootstrap_95_ci": {
            "top_k_average_net_return": {"low": net_ci_low},
            "top_k_average_spy_excess_return": {"low": benchmark_ci_low},
            "top_k_average_sector_excess_return": {"low": benchmark_ci_low},
        },
    }
    return {
        "candidate_id": candidate_id,
        "family": "logistic",
        "temporal_validation": metrics,
        "unseen_security_validation": metrics,
    }
