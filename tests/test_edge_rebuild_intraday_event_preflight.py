from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild import intraday_event_preflight as preflight
from market_predictor.core.errors import DataReadinessError


@dataclass
class _Harness:
    dataset_directory: Path
    event_directories: tuple[Path, Path]
    dataset: SimpleNamespace
    event_authorities: dict[Path, SimpleNamespace]
    config: preflight.IntradayEventPreflightConfig
    policy_path: Path

    def publish(self, output: Path, *, reverse_parents: bool = False) -> Any:
        parents = self.event_directories[::-1] if reverse_parents else self.event_directories
        arguments = {
            "dataset_authority_directory": self.dataset_directory,
            "event_authority_directories": parents,
            "output_directory": output,
            "config": self.config,
        }
        if "policy_path" in inspect.signature(
            preflight.publish_intraday_event_preflight
        ).parameters:
            arguments["policy_path"] = self.policy_path
        return preflight.publish_intraday_event_preflight(**arguments)


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    dataset_directory = tmp_path / "a43"
    dataset_directory.mkdir()
    _write_json(
        dataset_directory / "_authority.json",
        {"schema": "edge_rebuild.intraday_bar_dataset_authority.v1", "state": "complete"},
    )
    frame = _decision_frame()
    dataset = SimpleNamespace(
        frame=frame,
        root=dataset_directory,
        feature_columns=("stock_return_5m",),
        frozen_round_trip_cost_bps=10.0,
        dataset_sha256="1" * 64,
        manifest_sha256="1" * 64,
        authority_sha256=file_sha256(dataset_directory / "_authority.json"),
        request_sha256="2" * 64,
        transformation_sha256="3" * 64,
        session_unit_inventory_sha256="4" * 64,
        ordered_feature_sha256="5" * 64,
        strategy_contract_sha256="6" * 64,
    )

    event_authorities: dict[Path, SimpleNamespace] = {}
    event_directories: list[Path] = []
    sessions = sorted(frame["session_date_et"].astype(str).unique())
    for era, era_sessions in enumerate((sessions[:4], sessions[4:]), start=1):
        directory = tmp_path / f"events-era-{era}"
        directory.mkdir()
        _write_json(
            directory / "_authority.json",
            {
                "schema": "edge_rebuild.issuer_event_family_authority.v2",
                "state": "complete",
                "era": era,
            },
        )
        authority = _event_authority(
            directory,
            frame.loc[frame["session_date_et"].astype(str).isin(era_sessions)],
        )
        event_authorities[directory.resolve()] = authority
        event_directories.append(directory)

    def load_dataset(directory: Path) -> SimpleNamespace:
        assert Path(directory).resolve() == dataset_directory.resolve()
        return dataset

    def load_events(
        directory: Path,
        *,
        expected_authority_sha256: str | None = None,
    ) -> SimpleNamespace:
        resolved = Path(directory).resolve()
        authority = event_authorities[resolved]
        if expected_authority_sha256 is not None:
            assert expected_authority_sha256 == authority.authority_sha256
            assert expected_authority_sha256 == file_sha256(resolved / "_authority.json")
        return authority

    monkeypatch.setattr(preflight, "load_published_intraday_dataset", load_dataset)
    monkeypatch.setattr(preflight, "load_issuer_event_family_authority", load_events)
    policy_path = tmp_path / "intraday-event-preflight.toml"
    _write_policy(policy_path)
    return _Harness(
        dataset_directory=dataset_directory,
        event_directories=(event_directories[0], event_directories[1]),
        dataset=dataset,
        event_authorities=event_authorities,
        config=preflight.IntradayEventPreflightConfig(
            source_family="alpaca",
            relation_channel="direct_issuer",
            event_family="analyst_revision",
            lookback_hours=24,
            security_holdout_fraction=0.20,
            validation_folds=4,
            minimum_unique_event_episodes=1000,
            minimum_securities=200,
            minimum_fit_sessions=120,
            minimum_scope_rows=1000,
            minimum_scope_securities=20,
            maximum_process_memory_gib=4.0,
            memory_guard_headroom_gib=0.75,
        ),
        policy_path=policy_path,
    )


def test_proxy_only_evidence_publishes_blocked_fail_closed_output(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    for authority in harness.event_authorities.values():
        authority.events.loc[:, "availability_policy"] = "provider_publication_proxy"
        authority.events.loc[:, "production_eligible"] = False
        authority.coverage.loc[:, "production_eligible"] = False

    output = tmp_path / "proxy-blocked"
    harness.publish(output)
    loaded = preflight.load_intraday_event_preflight(output)
    manifest = _mapping(loaded, "manifest")
    authority = _mapping(loaded, "authority")

    assert _status(manifest, authority) == "blocked"
    assert manifest["training_eligible"] is False
    assert manifest["serving_eligible"] is False
    assert manifest["future_holdout_opened"] is False
    assert not any(
        path.suffix.lower() in {".joblib", ".pickle", ".pkl"}
        for path in output.rglob("*")
        if path.is_file()
    )
    for child_manifest in output.glob("*.parquet.manifest.json"):
        child = json.loads(child_manifest.read_text(encoding="utf-8"))
        assert child["production_ready"] is False
    decisions = _frame(loaded, "decision_eligibility", "decisions")
    assert not decisions["training_eligible"].astype(bool).any()
    assert "historical_availability_proxy_only" in manifest["blockers"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("lookback_hours", 12),
        ("security_holdout_fraction", 0.10),
        ("validation_folds", 2),
        ("minimum_unique_event_episodes", 999),
        ("minimum_securities", 199),
        ("minimum_fit_sessions", 119),
        ("minimum_scope_rows", 999),
        ("minimum_scope_securities", 19),
        ("maximum_process_memory_gib", 5.0),
    ),
)
def test_direct_config_cannot_weaken_frozen_policy(
    harness: _Harness,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    values = preflight._config_record(harness.config)
    values[field] = value
    harness.config = preflight.IntradayEventPreflightConfig(**values)

    with pytest.raises(DataReadinessError, match="frozen contract"):
        harness.publish(tmp_path / f"weakened-{field}")


def test_known_zero_and_unknown_coverage_remain_distinct(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    securities = sorted(harness.dataset.frame["security_id"].astype(str).unique())
    known_zero, unknown = securities[:2]
    first_decision = pd.to_datetime(
        harness.dataset.frame["decision_time_utc"], utc=True
    ).min()
    last_decision = pd.to_datetime(
        harness.dataset.frame["decision_time_utc"], utc=True
    ).max()
    for authority in harness.event_authorities.values():
        authority.events = authority.events.loc[
            ~authority.events["security_id"].astype(str).isin({known_zero, unknown})
        ].reset_index(drop=True)
        authority.assignments = authority.assignments.loc[
            ~authority.assignments["security_id"].astype(str).isin({known_zero, unknown})
        ].reset_index(drop=True)
        known_mask = authority.coverage["security_id"].astype(str).eq(known_zero)
        authority.coverage.loc[known_mask, "coverage_state"] = "observed_empty"
        authority.coverage.loc[known_mask, "missingness_known"] = True
        authority.coverage.loc[known_mask, "zero_event_semantics"] = "known_zero_events"
        authority.coverage.loc[known_mask, "requested_start_utc"] = first_decision - pd.Timedelta(
            hours=24
        )
        authority.coverage.loc[known_mask, "requested_end_utc"] = last_decision
        unknown_mask = authority.coverage["security_id"].astype(str).eq(unknown)
        authority.coverage.loc[unknown_mask, "coverage_state"] = "failed_or_unobserved"
        authority.coverage.loc[unknown_mask, "missingness_known"] = False
        authority.coverage.loc[unknown_mask, "zero_event_semantics"] = "unknown_failed"
        authority.coverage.loc[unknown_mask, "production_eligible"] = False

    output = tmp_path / "coverage-semantics"
    harness.publish(output)
    decisions = _frame(
        preflight.load_intraday_event_preflight(output),
        "decision_eligibility",
        "decisions",
    )
    known_rows = decisions.loc[decisions["security_id"].astype(str).eq(known_zero)]
    unknown_rows = decisions.loc[decisions["security_id"].astype(str).eq(unknown)]
    assert known_rows["production_coverage_state"].eq("known_zero_events").all()
    assert known_rows["production_event_count_24h"].eq(0).all()
    assert unknown_rows["production_coverage_state"].eq("unknown_or_proxy_only").all()
    assert unknown_rows["production_event_count_24h"].isna().all()
    assert not unknown_rows["training_eligible"].astype(bool).any()
    assert unknown_rows["ineligibility_reason"].astype(str).str.contains(
        "unknown", case=False
    ).all()


def test_late_collection_completion_cannot_create_historical_coverage(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    last_decision = pd.to_datetime(
        harness.dataset.frame["decision_time_utc"], utc=True
    ).max()
    for authority in harness.event_authorities.values():
        authority.coverage.loc[:, "completed_at_utc"] = last_decision + pd.Timedelta(
            days=1
        )

    result = harness.publish(tmp_path / "late-completion")
    decisions = _frame(result, "decisions")

    assert decisions["research_coverage_state"].eq("unknown").all()
    assert decisions["production_coverage_state"].eq(
        "unknown_or_proxy_only"
    ).all()
    assert decisions["research_event_count_24h"].isna().all()
    assert not decisions["training_eligible"].astype(bool).any()


def test_production_event_cannot_qualify_attachment_without_decision_coverage(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    last_decision = pd.to_datetime(
        harness.dataset.frame["decision_time_utc"], utc=True
    ).max()
    for authority in harness.event_authorities.values():
        authority.coverage.loc[:, "completed_at_utc"] = last_decision + pd.Timedelta(
            days=1
        )

    result = harness.publish(tmp_path / "ineligible-attachments")
    attachments = _frame(result, "attachments")

    assert attachments["production_eligible"].astype(bool).any()
    assert not attachments["attachment_eligible"].astype(bool).any()
    assert (
        attachments.loc[
            attachments["decision_id"].astype(str).ne(""), "attachment_status"
        ]
        .astype(str)
        .eq("attached_decision_ineligible")
        .all()
    )


def test_production_event_requires_revision_safe_observation(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    authority = next(iter(harness.event_authorities.values()))
    authority.events = authority.events.drop(columns=["revision_available_at_utc"])

    with pytest.raises(DataReadinessError, match="revision.*(lineage|timestamp)"):
        harness.publish(tmp_path / "missing-revision-lineage")


def test_issuer_identity_poison_is_rejected(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    authority = next(iter(harness.event_authorities.values()))
    authority.events.loc[0, "source_security_id"] = "security:wrong-issuer"

    with pytest.raises(DataReadinessError, match="issuer|security|identity"):
        harness.publish(tmp_path / "issuer-poison")


def test_ticker_and_compatible_cik_reconcile_historical_event_namespace(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    ticker = "T00"
    target_security_id = "cik:0000000001"
    source_security_id = f"{target_security_id}:ticker:{ticker}"
    decision_mask = harness.dataset.frame["ticker"].astype(str).eq(ticker)
    harness.dataset.frame.loc[decision_mask, "security_id"] = target_security_id
    for authority in harness.event_authorities.values():
        event_mask = authority.events["ticker"].astype(str).eq(ticker)
        authority.events.loc[event_mask, "security_id"] = source_security_id
        authority.events.loc[event_mask, "source_security_id"] = source_security_id
        coverage_mask = authority.coverage["ticker"].astype(str).eq(ticker)
        authority.coverage.loc[coverage_mask, "security_id"] = source_security_id
        assignment_mask = authority.assignments["ticker"].astype(str).eq(ticker)
        authority.assignments.loc[assignment_mask, "security_id"] = source_security_id

    result = harness.publish(tmp_path / "reconciled-namespace")
    attachments = _frame(result, "attachments")
    attached = attachments.loc[
        attachments["ticker"].astype(str).eq(ticker)
        & attachments["decision_id"].astype(str).ne("")
    ]

    assert not attached.empty
    assert attached["security_id"].astype(str).eq(target_security_id).all()
    assert attached["source_namespace_security_id"].astype(str).eq(
        source_security_id
    ).all()
    assert attached["identity_alignment"].astype(str).eq(
        "exact_ticker_cik_compatible"
    ).all()


def test_exact_ticker_with_conflicting_cik_is_rejected(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    ticker = "T00"
    decision_mask = harness.dataset.frame["ticker"].astype(str).eq(ticker)
    harness.dataset.frame.loc[decision_mask, "security_id"] = "cik:0000000001"
    authority = next(iter(harness.event_authorities.values()))
    event_mask = authority.events["ticker"].astype(str).eq(ticker)
    authority.events.loc[event_mask, "security_id"] = "cik:0000000002:ticker:T00"
    authority.events.loc[event_mask, "source_security_id"] = (
        "cik:0000000002:ticker:T00"
    )

    with pytest.raises(DataReadinessError, match="conflicting CIK"):
        harness.publish(tmp_path / "conflicting-cik")


def test_future_evidence_poison_is_rejected(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    authority = next(iter(harness.event_authorities.values()))
    event_id = str(authority.events.loc[0, "family_event_id"])
    decision_id = str(authority.assignments.loc[0, "decision_id"])
    decision_time = pd.Timestamp(authority.assignments.loc[0, "decision_time_utc"])
    authority.events.loc[0, "feature_available_at_utc"] = decision_time + pd.Timedelta(
        minutes=1
    )
    authority.assignments.loc[0, "feature_available_at_utc"] = decision_time + pd.Timedelta(
        minutes=1
    )

    assert event_id and decision_id
    with pytest.raises(DataReadinessError, match="future|decision|availability|point-in-time"):
        harness.publish(tmp_path / "future-poison")


def test_conflicting_duplicate_event_identity_is_rejected(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    authority = next(iter(harness.event_authorities.values()))
    duplicate = authority.events.iloc[[0]].copy()
    duplicate.loc[:, "feature_available_at_utc"] = pd.to_datetime(
        duplicate["feature_available_at_utc"], utc=True
    ) + pd.Timedelta(minutes=1)
    authority.events = pd.concat([authority.events, duplicate], ignore_index=True)

    with pytest.raises(DataReadinessError, match="duplicate|conflict|identity|repeat"):
        harness.publish(tmp_path / "duplicate-poison")


@pytest.mark.parametrize(
    "mutation", ("tamper", "manifest_tamper", "missing", "extra", "nested")
)
def test_strict_loader_rejects_tamper_and_inventory_drift(
    harness: _Harness,
    tmp_path: Path,
    mutation: str,
) -> None:
    output = tmp_path / f"strict-{mutation}"
    harness.publish(output)
    preflight.load_intraday_event_preflight(output)
    artifacts = sorted(path for path in output.rglob("*.parquet") if path.is_file())
    assert artifacts
    if mutation == "tamper":
        artifacts[0].write_bytes(artifacts[0].read_bytes() + b"tampered")
    elif mutation == "manifest_tamper":
        manifest_path = artifacts[0].with_suffix(".parquet.manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["created_at_utc"] = "2099-01-01T00:00:00+00:00"
        _write_json(manifest_path, manifest)
    elif mutation == "missing":
        artifacts[0].unlink()
    elif mutation == "extra":
        (output / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    else:
        nested = output / "unexpected"
        nested.mkdir()
        (nested / "payload.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(DataReadinessError, match="artifact|file|inventory|missing|unexpected"):
        preflight.load_intraday_event_preflight(output)


@pytest.mark.parametrize("input_name", ("dataset", "event", "policy"))
def test_publication_rejects_output_path_overlap(
    harness: _Harness,
    input_name: str,
) -> None:
    parent = (
        harness.dataset_directory
        if input_name == "dataset"
        else harness.event_directories[0] if input_name == "event" else harness.policy_path
    )

    with pytest.raises(DataReadinessError, match="overlap|inside|input|source"):
        output = parent / "overlapping-output" if parent.is_dir() else parent
        harness.publish(output)


def test_holdout_and_four_folds_are_deterministic_under_input_order(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    first_output = tmp_path / "first"
    harness.publish(first_output)
    first = _frame(
        preflight.load_intraday_event_preflight(first_output),
        "decision_eligibility",
        "decisions",
    )

    harness.dataset.frame = harness.dataset.frame.sample(frac=1.0, random_state=19).reset_index(
        drop=True
    )
    for authority in harness.event_authorities.values():
        authority.events = authority.events.sample(frac=1.0, random_state=23).reset_index(drop=True)
        authority.coverage = authority.coverage.sample(frac=1.0, random_state=29).reset_index(
            drop=True
        )
    second_output = tmp_path / "second"
    harness.publish(second_output, reverse_parents=True)
    second = _frame(
        preflight.load_intraday_event_preflight(second_output),
        "decision_eligibility",
        "decisions",
    )

    scope = _column(first, "validation_scope", "security_scope")
    fold = _column(first, "development_fold", "fold_id")
    first_scope = first.groupby("security_id", sort=True)[scope].first().astype(str)
    second_scope = second.groupby("security_id", sort=True)[scope].first().astype(str)
    pd.testing.assert_series_equal(first_scope, second_scope)
    expected_unseen = _stable_holdout(first_scope.index, 0.20)
    observed_unseen = set(first_scope.loc[first_scope.eq("unseen_security")].index.astype(str))
    assert observed_unseen == expected_unseen
    fold_values = set(pd.to_numeric(first[fold], errors="raise").dropna().astype(int))
    assert -1 in fold_values
    assert {value for value in fold_values if value >= 0} == {0, 1, 2, 3}
    first_attachments = _frame(
        preflight.load_intraday_event_preflight(first_output), "attachments"
    )
    second_attachments = _frame(
        preflight.load_intraday_event_preflight(second_output), "attachments"
    )
    pd.testing.assert_frame_equal(first_attachments, second_attachments)


def test_every_causal_event_decision_pair_is_published(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    duplicate_decision = harness.dataset.frame.iloc[[0]].copy()
    duplicate_decision.loc[:, "decision_id"] = "decision-extra-within-window"
    duplicate_decision.loc[:, "decision_time_utc"] = pd.to_datetime(
        duplicate_decision["decision_time_utc"], utc=True
    ) + pd.Timedelta(minutes=5)
    duplicate_decision.loc[:, "feature_available_at_utc"] = duplicate_decision[
        "decision_time_utc"
    ]
    harness.dataset.frame = pd.concat(
        [harness.dataset.frame, duplicate_decision], ignore_index=True
    )
    authority = harness.publish(tmp_path / "all-pairs")
    attachments = _frame(authority, "attachments")
    attached = attachments.loc[attachments["decision_id"].astype(str).ne("")]

    assert attached["family_event_id"].duplicated().any()
    assert (
        pd.to_datetime(attached["feature_available_at_utc"], utc=True)
        <= pd.to_datetime(attached["decision_time_utc"], utc=True)
    ).all()


def test_peak_memory_failure_prevents_publication(
    harness: _Harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "peak-memory-failure"
    monkeypatch.setattr(
        preflight,
        "assert_peak_memory_budget",
        lambda **_kwargs: (_ for _ in ()).throw(
            DataReadinessError("peak memory exceeded")
        ),
    )

    with pytest.raises(DataReadinessError, match="peak memory"):
        harness.publish(output)
    assert not output.exists()


def test_strict_loader_reuses_only_the_exact_verified_dataset(
    harness: _Harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "verified-dataset-reuse"
    harness.publish(output)
    monkeypatch.setattr(
        preflight,
        "load_published_intraday_dataset",
        lambda _directory: (_ for _ in ()).throw(
            AssertionError("verified parent must not be loaded twice")
        ),
    )

    preflight.load_intraday_event_preflight(
        output,
        verified_dataset=harness.dataset,
    )
    wrong_dataset = SimpleNamespace(
        **{
            **vars(harness.dataset),
            "root": tmp_path / "wrong-dataset",
        }
    )
    with pytest.raises(DataReadinessError, match="strict A4.3 parent replay"):
        preflight.load_intraday_event_preflight(
            output,
            verified_dataset=wrong_dataset,
        )


def _decision_frame() -> pd.DataFrame:
    securities = _fixture_securities()
    sessions = pd.bdate_range("2025-01-06", periods=124)
    rows: list[dict[str, object]] = []
    for session_index, session in enumerate(sessions):
        decision_time = pd.Timestamp(session.date(), tz="UTC") + pd.Timedelta(hours=15)
        for security_index, security_id in enumerate(securities):
            rows.append(
                {
                    "decision_id": f"decision-{session_index:02d}-{security_index:02d}",
                    "dataset_row_id": f"decision-{session_index:02d}-{security_index:02d}",
                    "decision_group_id": decision_time.isoformat(),
                    "security_id": security_id,
                    "ticker": f"T{security_index:02d}",
                    "decision_time_utc": decision_time,
                    "feature_available_at_utc": decision_time,
                    "session_date_et": session.date().isoformat(),
                    "sector": "Technology",
                    "strategy_contract_sha256": "6" * 64,
                    "feature_schema_version": "edge_rebuild.intraday_bar_features.v1",
                    "ordered_feature_sha256": "5" * 64,
                    "cost": 0.001,
                    "stock_return_5m": 0.01,
                }
            )
    return pd.DataFrame.from_records(rows)


def _event_authority(directory: Path, decisions: pd.DataFrame) -> SimpleNamespace:
    event_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for row in decisions.itertuples(index=False):
        available = pd.Timestamp(row.decision_time_utc) - pd.Timedelta(hours=1)
        event_id = f"event-{row.decision_id}"
        source_event_id = f"provider-{row.decision_id}"
        event_rows.append(
            {
                "family_event_id": event_id,
                "source_event_id": source_event_id,
                "relation_id": f"relation-{row.decision_id}",
                "source_security_id": row.security_id,
                "source_ticker": row.ticker,
                "security_id": row.security_id,
                "ticker": row.ticker,
                "source_family": "alpaca",
                "event_family": "analyst_revision",
                "published_at_utc": available - pd.Timedelta(minutes=5),
                "event_available_at_utc": available,
                "relation_available_at_utc": available,
                "first_seen_at_utc": available,
                "revision_id": f"revision-{source_event_id}-1",
                "revision_available_at_utc": available,
                "feature_available_at_utc": available,
                "availability_policy": "observed",
                "relation_channel": "direct_issuer",
                "relation_score": 1.0,
                "research_eligible": True,
                "production_eligible": True,
                "exclusion_reason": "",
            }
        )
        assignment_rows.append(
            {
                "decision_id": row.decision_id,
                "security_id": row.security_id,
                "ticker": row.ticker,
                "decision_time_utc": row.decision_time_utc,
                "event_id": event_id,
                "family_event_id": event_id,
                "source_event_id": source_event_id,
                "feature_available_at_utc": available,
                "window_name": "1d",
                "status": "assigned",
                "event_family": "analyst_revision",
                "original_source_family": "alpaca",
            }
        )
        coverage_rows.append(
            {
                "collection_id": f"collection-{row.decision_id}",
                "chunk_id": f"chunk-{row.decision_id}",
                "security_id": row.security_id,
                "ticker": row.ticker,
                "source_family": "alpaca",
                "event_family": "analyst_revision",
                "requested_start_utc": pd.Timestamp(row.decision_time_utc)
                - pd.Timedelta(hours=24),
                "requested_end_utc": row.decision_time_utc,
                "completed_at_utc": pd.Timestamp(row.decision_time_utc)
                - pd.Timedelta(seconds=1),
                "coverage_state": "observed_complete",
                "missingness_known": True,
                "zero_event_semantics": "observed_history",
                "research_eligible": True,
                "production_eligible": True,
            }
        )
    authority_sha256 = file_sha256(directory / "_authority.json")
    return SimpleNamespace(
        directory=directory,
        events=pd.DataFrame.from_records(event_rows),
        assignments=pd.DataFrame.from_records(assignment_rows),
        coverage=pd.DataFrame.from_records(coverage_rows),
        cohort_audit=pd.DataFrame(),
        manifest={
            "state": "complete",
            "production_ready": True,
            "event_family_policy_sha256": "7" * 64,
            "family_status": {"analyst_revision": "admitted"},
        },
        authority={"state": "complete", "production_ready": True},
        authority_sha256=authority_sha256,
        projected_inventory_sha256=hashlib.sha256(
            str(directory.resolve()).encode("utf-8")
        ).hexdigest(),
    )


def _fixture_securities() -> tuple[str, ...]:
    candidates = [f"security:{index:03d}" for index in range(200)]
    held = sorted(_stable_holdout(candidates, 0.20))
    seen = [security for security in candidates if security not in held]
    assert len(held) >= 4 and len(seen) >= 12
    return tuple(sorted([*held[:4], *seen[:12]]))


def _stable_holdout(securities: Any, fraction: float) -> set[str]:
    threshold = int(fraction * 2**64)
    return {
        str(security)
        for security in securities
        if int(hashlib.sha256(str(security).encode("utf-8")).hexdigest()[:16], 16)
        < threshold
    }


def _frame(value: object, *names: str) -> pd.DataFrame:
    for name in names:
        candidate = value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
        if isinstance(candidate, pd.DataFrame):
            return candidate
    raise AssertionError(f"preflight result does not expose any of {names}")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    candidate = value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
    if not isinstance(candidate, Mapping):
        raise AssertionError(f"preflight result does not expose mapping {name}")
    return candidate


def _column(frame: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise AssertionError(f"preflight evidence does not expose any of {names}")


def _status(
    manifest: Mapping[str, object],
    authority: Mapping[str, object],
) -> str:
    return str(authority.get("status", authority.get("state", manifest.get("status", ""))))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _write_policy(path: Path) -> None:
    path.write_text(
        """schema_version = "edge_rebuild.intraday_event_preflight_policy.v1"
source_family = "alpaca"
relation_channel = "direct_issuer"
event_family = "analyst_revision"
lookback_hours = 24
security_holdout_fraction = 0.20
validation_folds = 4
minimum_unique_event_episodes = 1000
minimum_securities = 200
minimum_fit_sessions = 120
minimum_scope_rows = 1000
minimum_scope_securities = 20
maximum_process_memory_gib = 4.0
memory_guard_headroom_gib = 0.75
unknown_coverage_policy = "abstain"
historical_proxy_policy = "research_only"
""",
        encoding="utf-8",
    )
